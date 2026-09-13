
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Ganta Planning Block Generator", layout="wide")

DATA = "Ganta_BaseData.gpkg"
WARDS = "Ganta_Wards.shp"

@st.cache_data
def load_data():
    roads = gpd.read_file(DATA, layer="ganta_roads")
    buildings = gpd.read_file(DATA, layer="ganta_buildings")
    wards = gpd.read_file(WARDS)
    # Standardize the ward label field used by the map tooltip.
    # The source ward layer contains Id and Zone_Code, not a field named ward.
    wards["ward"] = wards["Zone_Code"].fillna(wards["Id"]).astype(str)
    roads = roads.to_crs(wards.crs)
    buildings = buildings.to_crs(wards.crs)
    return roads, buildings, wards

def clean_geom(g):
    try:
        g = g.buffer(0)
    except Exception:
        pass
    return g

def road_class_mask(roads, classes):
    return roads[roads["highway"].isin(classes)].copy()

def polygonize_ward(ward_geom, roads):
    """Create cells from the selected road network inside one ward."""
    clipped = []
    for geom in roads.geometry:
        if geom is None or geom.is_empty:
            continue
        inter = geom.intersection(ward_geom)
        if inter.is_empty:
            continue
        if inter.geom_type == "LineString":
            clipped.append(inter)
        elif inter.geom_type == "MultiLineString":
            clipped.extend(list(inter.geoms))
    # Ward boundary closes the network.
    boundary = ward_geom.boundary
    lines = clipped + list(boundary.geoms) if boundary.geom_type == "MultiLineString" else clipped + [boundary]
    if not lines:
        return [ward_geom]
    try:
        polys = list(polygonize(unary_union(lines)))
    except Exception:
        return [ward_geom]
    cells = []
    for p in polys:
        q = p.intersection(ward_geom)
        if not q.is_empty and q.area > 1:
            cells.append(clean_geom(q))
    return cells or [ward_geom]

def adjacency(cells):
    pairs = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            try:
                shared = cells[i].boundary.intersection(cells[j].boundary).length
            except Exception:
                shared = 0
            if shared > 1.0:
                pairs.append((i, j, shared))
    return pairs


def split_polygon_balanced(poly, target):
    """Split a polygon robustly using the longest bounding-box dimension.
    MultiPolygons are handled part-by-part. The cut is an artificial planning
    line used only to prevent oversized blocks.
    """
    poly = clean_geom(poly)
    if poly.is_empty or poly.area <= target * 1.05:
        return [poly]

    if poly.geom_type == "MultiPolygon":
        pieces = []
        for part in poly.geoms:
            pieces.extend(split_polygon_balanced(part, target))
        return pieces

    minx, miny, maxx, maxy = poly.bounds
    width, height = maxx - minx, maxy - miny
    use_x = width >= height

    def cut_at(v):
        pad = max(width, height) * 2 + 10
        if use_x:
            line = LineString([(v, miny-pad), (v, maxy+pad)])
        else:
            line = LineString([(minx-pad, v), (maxx+pad, v)])
        try:
            from shapely.ops import split
            result = split(poly, line)
            return [clean_geom(g) for g in result.geoms if g.area > 1]
        except Exception:
            return []

    # Find a cut close to a half-area division. This guarantees recursive
    # subdivision rather than leaving a huge road cell untouched.
    lo, hi = (minx, maxx) if use_x else (miny, maxy)
    best = []
    best_err = float('inf')
    for _ in range(36):
        mid = (lo + hi) / 2
        pieces = cut_at(mid)
        if len(pieces) < 2:
            # If the line does not cross this geometry, move through the range.
            if use_x:
                lo = mid
            else:
                lo = mid
            continue
        # For a simple cut, compare the largest resulting side with half.
        pieces_sorted = sorted(pieces, key=lambda g: g.area, reverse=True)
        a = pieces_sorted[0].area
        b = sum(g.area for g in pieces_sorted[1:])
        err = abs(a - b)
        if err < best_err:
            best_err, best = err, pieces
        if a > b:
            if use_x: hi = mid
            else: hi = mid
        else:
            if use_x: lo = mid
            else: lo = mid

    if len(best) < 2:
        # Try several quantile positions as a fallback for irregular geometry.
        for frac in (0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75):
            v = lo + (hi-lo)*frac
            pieces = cut_at(v)
            if len(pieces) >= 2:
                best = pieces
                break

    if len(best) < 2:
        return [poly]

    # If a cut yields more than two pieces, assign fragments to the nearest
    # principal piece so the output remains a clean set of non-overlapping blocks.
    if len(best) > 2:
        best = sorted(best, key=lambda g: g.area, reverse=True)
        a, b = best[0], best[1]
        for frag in best[2:]:
            if frag.centroid.distance(a) <= frag.centroid.distance(b):
                a = clean_geom(a.union(frag))
            else:
                b = clean_geom(b.union(frag))
        best = [a, b]
    return best


def subdivide_oversized(cells, target, tolerance):
    """Recursively subdivide cells until no block is extremely oversized."""
    result = []
    queue = list(cells)
    max_area = target * (1 + tolerance)
    guard = 0
    while queue and guard < 10000:
        guard += 1
        poly = clean_geom(queue.pop())
        if poly.is_empty or poly.area <= 1:
            continue
        if poly.area <= max_area:
            result.append(poly)
            continue
        pieces = split_polygon_balanced(poly, target)
        if len(pieces) < 2:
            # Last-resort bounding-box slicing using several candidate cuts.
            minx, miny, maxx, maxy = poly.bounds
            width, height = maxx-minx, maxy-miny
            candidates = []
            if width >= height:
                for frac in (0.33, 0.40, 0.50, 0.60, 0.67):
                    x = minx + width*frac
                    line = LineString([(x,miny-10*max(height,1)),(x,maxy+10*max(height,1))])
                    try:
                        from shapely.ops import split
                        pp=[clean_geom(g) for g in split(poly,line).geoms if g.area>1]
                        if len(pp)>=2: candidates.append(pp)
                    except Exception: pass
            else:
                for frac in (0.33, 0.40, 0.50, 0.60, 0.67):
                    y = miny + height*frac
                    line = LineString([(minx-10*max(width,1),y),(maxx+10*max(width,1),y)])
                    try:
                        from shapely.ops import split
                        pp=[clean_geom(g) for g in split(poly,line).geoms if g.area>1]
                        if len(pp)>=2: candidates.append(pp)
                    except Exception: pass
            if candidates:
                pieces=min(candidates,key=lambda pp:max(g.area for g in pp))
            else:
                result.append(poly); continue
        # Keep all resulting pieces; recursively process each.
        queue.extend(pieces)
    return result


def merge_small_and_uneven(cells, target, min_area, tolerance):
    """Merge small cells only where the merged result remains reasonably sized."""
    cells = [clean_geom(c) for c in cells if c.area > 1]
    low = max(min_area, target * 0.35)
    high = target * (1 + tolerance)

    while len(cells) > 1:
        small = [i for i,c in enumerate(cells) if c.area < low]
        if not small:
            break
        i = min(small, key=lambda k: cells[k].area)
        best = None; best_score = float('inf')
        for j in range(len(cells)):
            if j == i: continue
            try: shared=cells[i].boundary.intersection(cells[j].boundary).length
            except Exception: shared=0
            if shared <= 1: continue
            new_area=cells[i].area+cells[j].area
            # Never merge into an extreme block.
            if new_area > high: continue
            score=abs(new_area-target) - min(shared/100.0, 30.0)
            if score < best_score:
                best_score,best=score,j
        if best is None:
            break
        merged=clean_geom(cells[i].union(cells[best]))
        cells=[c for k,c in enumerate(cells) if k not in (i,best)]
        cells.append(merged)
    return cells


def final_balance(cells, target, tolerance, min_area):
    # Use a firm ceiling to prevent the very large blocks the earlier version allowed.
    cells = subdivide_oversized(cells, target, tolerance)
    cells = merge_small_and_uneven(cells, target, min_area, tolerance)
    cells = subdivide_oversized(cells, target, tolerance)
    return [clean_geom(c) for c in cells if c.area > 1]

def partition_nonoverlap(cells, ward_geom):
    """Create a true partition from candidate cells: no overlaps or nested blocks."""
    cells = [clean_geom(c.intersection(ward_geom)) for c in cells if c is not None and not c.is_empty]
    cells = [c for c in cells if c.area > 1]
    if not cells:
        return [clean_geom(ward_geom)]
    lines = [ward_geom.boundary] + [c.boundary for c in cells]
    try:
        faces = [f for f in polygonize(unary_union(lines)) if f.area > 1]
    except Exception:
        return cells
    assigned = [None] * len(cells)
    for face in faces:
        best_i, best_a = None, 0.0
        for i, c in enumerate(cells):
            a = face.intersection(c).area
            if a > best_a:
                best_i, best_a = i, a
        if best_i is not None:
            assigned[best_i] = face if assigned[best_i] is None else assigned[best_i].union(face)
    out=[]
    for g in assigned:
        if g is not None:
            g=clean_geom(g.intersection(ward_geom))
            if not g.is_empty and g.area > 1:
                out.append(g)
    return out

def generate(target_m2, tolerance, min_area, classes):
    roads, buildings, wards = load_data()
    selected = road_class_mask(roads, classes)

    all_blocks = []
    for _, w in wards.iterrows():
        wg = clean_geom(w.geometry)
        r = selected[selected.intersects(wg)]
        cells = polygonize_ward(wg, r)

        # Road cells are the first framework. Oversized cells are then subdivided by
        # clean, single cut lines; this prevents huge areas when roads are sparse.
        cells = final_balance(cells, target_m2, tolerance, min_area)

        # If roads still leave an unusually large cell, add lower-order roads only as
        # a second-pass framework, then run the same equalisation process.
        if cells and max(c.area for c in cells) > target_m2 * (1 + tolerance):
            lower_order = ["residential", "unclassified", "service", "track"]
            classes2 = list(classes)
            for cls in lower_order:
                if cls not in classes2:
                    classes2.append(cls)
                rr = roads[roads["highway"].isin(classes2)]
                rr = rr[rr.intersects(wg)]
                trial = polygonize_ward(wg, rr)
                trial = final_balance(trial, target_m2, tolerance, min_area)
                if len(trial) > 0 and max(c.area for c in trial) < max(x.area for x in cells):
                    cells = trial
                if cells and max(c.area for c in cells) <= target_m2 * (1 + tolerance):
                    break

        # Hard guarantee: no block-in-block and no overlapping blocks.
        cells = partition_nonoverlap(cells, wg)
        for c in cells:
            c = clean_geom(c.intersection(wg))
            if not c.is_empty and c.area > 1:
                all_blocks.append({"ward": str(w.get("ward", w.get("Zone_Code", w.get("Id", "")))), "geometry": c})

    blocks = gpd.GeoDataFrame(all_blocks, crs=wards.crs)

    # Validation indicators from building centroids.
    bpts = buildings.copy()
    bpts["geometry"] = bpts.geometry.centroid
    joined = gpd.sjoin(bpts, blocks[["ward", "geometry"]], predicate="within", how="left")
    counts = joined.groupby("index_right").size()
    blocks["building_count"] = [int(counts.get(i, 0)) for i in blocks.index]
    blocks["area_m2"] = blocks.area
    blocks["area_ha"] = blocks.area / 10000
    blocks["target_m2"] = target_m2
    blocks["deviation_pct"] = (blocks.area / target_m2 - 1) * 100
    blocks["status"] = np.select(
        [blocks.area < target_m2 * (1 - tolerance),
         blocks.area > target_m2 * (1 + tolerance)],
        ["undersized", "oversized"], default="within_tolerance"
    )
    blocks["block_id"] = [f"{w}-B{i:03d}" for i, w in enumerate(blocks.ward, 1)]
    blocks = blocks[["block_id", "ward", "area_m2", "area_ha", "target_m2", "deviation_pct", "status", "building_count", "geometry"]]
    return blocks, wards, roads, buildings

st.title("Ganta Parametric Planning Block Generator")
st.caption("Road-based planning blocks constrained by ward boundaries. Target area is an optimization target, not an exact legal parcel size.")

with st.sidebar:
    st.header("Block parameters")
    target_ha=st.number_input("Target block area (ha)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    tolerance_pct=st.slider("Allowed variation (%)", 5, 50, 20, 5)
    min_ha=st.number_input("Minimum block area (ha)", min_value=0.25, max_value=50.0, value=5.0, step=0.25)
    st.subheader("Road hierarchy")
    major=st.multiselect("Start with roads", ["primary","trunk","tertiary","unclassified","residential","service","track","path"],
                         default=["primary","trunk","tertiary","unclassified","residential"])
    st.subheader("Map styling")
    ward_color=st.color_picker("Ward outline colour", "#222222")
    ward_weight=st.slider("Ward line weight", 1.0, 8.0, 4.0, 0.5)
    block_color=st.color_picker("Block outline colour", "#0066CC")
    block_weight=st.slider("Block line weight", 0.5, 8.0, 2.5, 0.5)
    block_fill=st.checkbox("Fill planning blocks", value=False)
    block_fill_opacity=st.slider("Block fill opacity", 0.0, 0.8, 0.15, 0.05)
    generate_btn=st.button("Generate blocks", type="primary", use_container_width=True)
    st.info("Tip: start around 20 ha target and ±20% tolerance. Add track/path only if large areas remain unbroken.")

if "blocks" not in st.session_state or generate_btn:
    with st.spinner("Generating road-based planning blocks..."):
        blocks, wards, roads, buildings = generate(target_ha*10000, tolerance_pct/100, min_ha*10000, major)
        st.session_state.blocks=blocks
        st.session_state.wards=wards
        st.session_state.roads=roads
        st.session_state.buildings=buildings

blocks=st.session_state.blocks
wards=st.session_state.wards
roads=st.session_state.roads

# Folium/Leaflet expects geographic coordinates (longitude/latitude).
# The Ganta source data are in UTM EPSG:32629, so reproject map layers first.
map_blocks = blocks.to_crs(epsg=4326)
map_wards = wards.to_crs(epsg=4326)
map_roads = roads.to_crs(epsg=4326)

c1,c2,c3,c4=st.columns(4)
c1.metric("Planning blocks",len(blocks))
c2.metric("Target area",f"{blocks.target_m2.iloc[0]/10000:.1f} ha")
c3.metric("Within tolerance",f"{(blocks.status=='within_tolerance').mean()*100:.0f}%")
c4.metric("Median block",f"{blocks.area_ha.median():.1f} ha")

left,right=st.columns([2,1])
with left:
    # Folium/Leaflet expects geographic coordinates (longitude/latitude).
    # Ganta source data are in UTM EPSG:32629, so reproject map layers first.
    map_blocks = blocks.to_crs(epsg=4326)
    map_wards = wards.to_crs(epsg=4326)
    map_roads = roads.to_crs(epsg=4326)

    # Start from the actual ward extent rather than a hard-coded center.
    bounds = map_wards.total_bounds  # minx, miny, maxx, maxy
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Give each planning block a distinct fill colour while keeping ward boundaries
    # visually separate.  The tooltip on each block shows both its block ID and ward.

    # Ward polygons are transparent so the coloured planning blocks remain visible.
    # Hovering a ward shows the ward label.
    folium.GeoJson(
        map_wards.to_json(),
        name="Wards",
        style_function=lambda x: {
            "fillOpacity": 0.0,
            "color": ward_color,
            "weight": ward_weight,
            "dashArray": "8, 5"
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ward"],
            aliases=["Ward"],
            sticky=False
        )
    ).add_to(m)

    folium.GeoJson(
        map_blocks.to_json(),
        name="Planning Blocks",
        style_function=lambda feature: {
            "fillOpacity": block_fill_opacity if block_fill else 0.0,
            "fillColor": block_color,
            "color": block_color,
            "weight": block_weight,
            "opacity": 0.95
        },
        highlight_function=lambda feature: {
            "weight": max(block_weight + 1.5, block_weight),
            "fillOpacity": block_fill_opacity if block_fill else 0.0,
            "fillColor": block_color
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["block_id", "ward", "area_ha", "status", "building_count"],
            aliases=["Block", "Ward", "Area (ha)", "Status", "Buildings"],
            sticky=False
        )
    ).add_to(m)

    folium.GeoJson(
        map_roads.to_json(),
        name="Roads",
        style_function=lambda x: {
            "weight": 1.5,
            "opacity": 0.65
        }
    ).add_to(m)

    # Always zoom to the complete Ganta ward extent.
    m.fit_bounds([
        [bounds[1], bounds[0]],
        [bounds[3], bounds[2]]
    ])
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, height=650, width=None)

with right:
    st.subheader("Area distribution")
    st.dataframe(
        blocks.drop(columns="geometry").sort_values(["ward", "area_m2"]),
        use_container_width=True,
        height=520
    )
    st.download_button(
        "Download planning blocks (GeoJSON)",
        blocks.to_json(),
        "Ganta_PlanningBlocks.geojson",
        "application/geo+json"
    )
    st.download_button(
        "Download summary (CSV)",
        blocks.drop(columns="geometry").to_csv(index=False),
        "Ganta_PlanningBlocks_Summary.csv",
        "text/csv"
    )

st.markdown("### Planning rule used")
st.write(
    "Ward boundary = hard boundary; selected roads = preferred/hard subdivision lines; "
    "blocks are merged where possible to approach the target area. Buildings are used "
    "only as a validation indicator, not as the block boundary."
)
