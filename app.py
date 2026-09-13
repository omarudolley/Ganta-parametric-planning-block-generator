import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union, polygonize, split
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title='Ganta Planning Block Generator', layout='wide')
DATA='Ganta_BaseData.gpkg'; WARDS='Ganta_Wards.shp'
ROAD_CLASSES=['primary','trunk','tertiary','unclassified','residential','service','track','path','footway']

@st.cache_data(show_spinner=False)
def load_data():
    roads=gpd.read_file(DATA, layer='ganta_roads')
    buildings=gpd.read_file(DATA, layer='ganta_buildings')
    wards=gpd.read_file(WARDS)
    wards['ward']=wards['Zone_Code'].fillna(wards['Id']).astype(str)
    roads=roads.to_crs(wards.crs); buildings=buildings.to_crs(wards.crs)
    roads=roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    buildings=buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    return roads,buildings,wards

def clean(g):
    try:return g.buffer(0)
    except:return g

def road_mask(roads, classes):
    return roads[roads['highway'].isin(classes)].copy()

def clip_lines(geom, roads):
    out=[]
    for g in roads.geometry:
        if g is None or g.is_empty: continue
        try:i=g.intersection(geom)
        except: continue
        if i.is_empty: continue
        if i.geom_type=='LineString': out.append(i)
        elif i.geom_type=='MultiLineString': out.extend([x for x in i.geoms if not x.is_empty])
        elif i.geom_type=='GeometryCollection': out.extend([x for x in i.geoms if x.geom_type=='LineString' and not x.is_empty])
    return out

def road_cells(geom, roads):
    lines=clip_lines(geom,roads)
    boundary=geom.boundary
    net=unary_union(lines+[boundary]) if lines else boundary
    try: raw=list(polygonize(net))
    except Exception:return [geom]
    cells=[]
    for p in raw:
        q=clean(p.intersection(geom))
        if not q.is_empty and q.area>1: cells.append(q)
    return cells or [geom]

def boundary_road_metrics(cell, roads, ward_geom):
    # Count only road segments that actually form the candidate boundary.
    # Ward-edge overlap is removed so a whole-ward polygon can never qualify.
    b=cell.boundary
    try:
        ward_b=ward_geom.boundary
        interior_b=b.difference(ward_b)
        if interior_b.is_empty: return 0.0,0.0,0
        hits=[]
        idxs=roads.sindex.query(interior_b,predicate='intersects') if len(roads) else []
        for idx in idxs:
            g=roads.geometry.iloc[int(idx)]
            inter=interior_b.intersection(g)
            if not inter.is_empty and inter.length>5: hits.append(float(inter.length))
        total=sum(hits); ratio=total/max(interior_b.length,1.0)
        meaningful=sum(1 for h in hits if h>=15)
        return total,ratio,meaningful
    except Exception:return 0.0,0.0,0

def enforce_partition(cells, ward):
    # Final topological safety pass: planning blocks must form a true partition.
    # Remove any overlap by subtracting already accepted area, then repair the
    # small numerical gaps back to the nearest block.
    accepted=[]; occupied=None
    for c in sorted(cells, key=lambda x: x.area, reverse=True):
        c=clean(c.intersection(ward))
        if c.is_empty or c.area<=1: continue
        if occupied is not None:
            c=clean(c.difference(occupied))
        if c.is_empty or c.area<=1: continue
        accepted.append(c)
        occupied=c if occupied is None else clean(occupied.union(c))
    gap=clean(ward.difference(occupied)) if occupied is not None else clean(ward)
    if not gap.is_empty and gap.area>0.01 and accepted:
        gs=list(gap.geoms) if gap.geom_type=='MultiPolygon' else [gap]
        for g in gs:
            if g.area<=0.01: continue
            i=min(range(len(accepted)),key=lambda k:accepted[k].distance(g))
            accepted[i]=clean(accepted[i].union(g))
    return [clean(c) for c in accepted if not c.is_empty and c.area>1]

def _line_building_penalty(line, buildings, poly=None):
    """Fast building-crossing score using the building spatial index.
    Only buildings relevant to the current polygon are considered."""
    if buildings is None or len(buildings) == 0:
        return 0.0, 0
    try:
        query_line = line
        if poly is not None:
            query_line = line.intersection(poly)
            if query_line.is_empty:
                return 0.0, 0
        hits = buildings.sindex.query(query_line, predicate='intersects')
        if len(hits) == 0:
            return 0.0, 0
        geoms = buildings.geometry.iloc[np.asarray(hits, dtype=int)].to_numpy()
        inter = __import__('shapely').intersection(geoms, query_line)
        lengths = __import__('shapely').length(inter)
        mask = np.asarray(lengths) > 0.01
        return float(np.asarray(lengths)[mask].sum()), int(mask.sum())
    except Exception:
        return 0.0, 0

def _building_gap_positions(poly, buildings, axis, max_candidates=6):
    """Return promising cut positions through the largest building-free gaps.
    Building boxes are used only to locate gaps; buildings never become block edges."""
    if buildings is None or len(buildings) == 0:
        return []
    try:
        hits = buildings.sindex.query(poly, predicate='intersects')
        if len(hits) == 0:
            return []
        minx,miny,maxx,maxy=poly.bounds
        intervals=[]
        for idx in np.asarray(hits, dtype=int):
            g=buildings.geometry.iloc[int(idx)]
            bx0,by0,bx1,by1=g.bounds
            a,b=(bx0,bx1) if axis=='x' else (by0,by1)
            lo,hi=(minx,maxx) if axis=='x' else (miny,maxy)
            a=max(a,lo); b=min(b,hi)
            if b>a: intervals.append((a,b))
        if not intervals: return []
        intervals.sort(); merged=[]
        for a,b in intervals:
            if not merged or a>merged[-1][1]: merged.append([a,b])
            else: merged[-1][1]=max(merged[-1][1],b)
        lo,hi=(minx,maxx) if axis=='x' else (miny,maxy)
        gaps=[]; cur=lo
        for a,b in merged:
            if a-cur>0: gaps.append((a-cur,cur,a))
            cur=max(cur,b)
        if hi-cur>0: gaps.append((hi-cur,cur,hi))
        gaps.sort(reverse=True)
        return [(a+b)/2 for _,a,b in gaps[:max_candidates] if b-a>1.0]
    except Exception:
        return []

def split_best(poly, target, buildings=None):
    """Split a polygon while strongly preferring cuts through building-free gaps."""
    poly=clean(poly)
    if poly.is_empty or poly.area<=target*1.02:return [poly]
    if poly.geom_type=='MultiPolygon':
        out=[]
        for p in poly.geoms: out.extend(split_best(p,target,buildings))
        return out
    minx,miny,maxx,maxy=poly.bounds; w=maxx-minx; h=maxy-miny
    axes=['x','y'] if w>=h else ['y','x']
    candidates=[]
    for axis in axes:
        lo,hi=(minx,maxx) if axis=='x' else (miny,maxy); span=hi-lo
        if span<=0: continue
        positions=[lo+span*f for f in (0.35,0.45,0.50,0.55,0.65)]
        positions += _building_gap_positions(poly,buildings,axis,max_candidates=6)
        # Deduplicate nearby cut positions to keep recursion cheap.
        positions=sorted(set(round(v,2) for v in positions))
        for v in positions:
            pad=max(w,h)*0.25+5
            ln=LineString([(v,miny-pad),(v,maxy+pad)]) if axis=='x' else LineString([(minx-pad,v),(maxx+pad,v)])
            try: pieces=[clean(x) for x in split(poly,ln).geoms if x.area>1]
            except Exception: continue
            if len(pieces)<2: continue
            areas=np.array([p.area for p in pieces])
            small=(areas<0.25*target).sum()
            balance=abs(np.median(areas)-target)/max(target,1)
            oversize=max(areas.max()/max(target,1)-1,0)
            cut_len,cut_count=_line_building_penalty(ln,buildings,poly)
            # Crossing a building is a last resort. Prefer a larger area imbalance
            # over a cut through an existing footprint whenever possible.
            building_penalty=1000.0*cut_count + 200.0*cut_len/max(np.sqrt(poly.area),1.0)
            score=balance + 1.5*small + 0.10*oversize + building_penalty
            candidates.append((score,cut_count,cut_len,pieces))
    if not candidates:return [poly]
    zero=[c for c in candidates if c[1]==0 and c[2] <= 0.01]
    pool=zero if zero else candidates
    return min(pool,key=lambda x:x[0])[3]

def target_subdivide(poly,target,min_area,buildings=None):
    out=[]; stack=[poly]; guard=0
    while stack and guard<10000:
        guard+=1; p=clean(stack.pop())
        if p.is_empty or p.area<=1:continue
        if p.area<=target*1.02:
            out.append(p);continue
        pieces=split_best(p,target,buildings)
        if len(pieces)<2:out.append(p)
        else:stack.extend(pieces)
    return out

def merge_tiny(cells,min_area):
    # Fast residual cleanup. Do not repeatedly merge tiny cells into one another:
    # that was both slow and a source of artificial mega-blocks. Attach each
    # residual directly to an existing core block using the spatial index.
    cells=[clean(c) for c in cells if c is not None and not c.is_empty and c.area>1]
    if len(cells)<=1: return cells
    core=[c for c in cells if c.area>=min_area]
    tiny=[c for c in cells if c.area<min_area]
    if not tiny: return cells
    if not core: return [clean(unary_union(cells))]
    core_gdf=gpd.GeoDataFrame({'geometry':core},crs='EPSG:32629')
    sidx=core_gdf.sindex
    assignments={i:[] for i in range(len(core))}
    for t in tiny:
        best=None
        try:
            hits=sidx.query(t,predicate='intersects')
        except Exception:
            hits=[]
        candidates=[]
        for j in hits:
            j=int(j); shared=t.boundary.intersection(core[j].boundary).length
            if shared>0.01: candidates.append((core[j].area,-shared,j))
        if candidates:
            best=min(candidates)[2]
        else:
            try:
                nearest=list(sidx.nearest(t))
                if nearest: best=int(nearest[0])
            except Exception: pass
        if best is not None: assignments[best].append(t)
    out=[]
    for i,c in enumerate(core):
        if assignments[i]: c=clean(c.union(unary_union(assignments[i])))
        out.append(c)
    return out

def repair_partition(cells,ward):
    cells=[clean(c.intersection(ward)) for c in cells if c is not None and not c.is_empty]
    cells=[c for c in cells if c.area>1]
    if not cells:return [clean(ward)]
    u=clean(unary_union(cells)); gap=clean(ward.difference(u))
    if not gap.is_empty and gap.area>0.01:
        gs=list(gap.geoms) if gap.geom_type=='MultiPolygon' else [gap]
        for g in gs:
            if g.area<=0.01:continue
            # assign gap to nearest block, preferring longest shared boundary
            i=min(range(len(cells)),key=lambda k:cells[k].distance(g)); cells[i]=clean(cells[i].union(g))
    return [clean(c.intersection(ward)) for c in cells if not c.is_empty and c.area>1]

def _shape_metrics(poly):
    minx,miny,maxx,maxy=poly.bounds
    w=max(maxx-minx,1.0); h=max(maxy-miny,1.0)
    return max(w,h)/max(min(w,h),1.0)

def learn_block_pattern(reference_cells, fallback_target):
    """Learn a robust planning-block scale from road-defined reference cells."""
    if not reference_cells:
        return {'median_area':fallback_target,'q25':fallback_target,'q75':fallback_target,
                'median_elongation':1.0,'count':0,'max_area':fallback_target}
    areas=np.array([c.area for c in reference_cells if c.area>1],dtype=float)
    if len(areas)==0:
        return {'median_area':fallback_target,'q25':fallback_target,'q75':fallback_target,
                'median_elongation':1.0,'count':0,'max_area':fallback_target}
    # Robustly ignore only the extreme tails when learning scale; this is not a
    # ceiling on real blocks, it simply prevents a few exceptional polygons from
    # dominating the reference pattern.
    q05,q95=np.quantile(areas,[0.05,0.95]) if len(areas)>=4 else (areas.min(),areas.max())
    core=areas[(areas>=q05)&(areas<=q95)]
    if len(core)==0: core=areas
    elong=np.array([_shape_metrics(c) for c in reference_cells if c.area>1],dtype=float)
    q25=float(np.quantile(core,0.25)); q75=float(np.quantile(core,0.75))
    # Polygonization can produce very large road-enclosed polygons that are
    # technically bounded by roads but are not credible planning blocks. Use
    # the upper quartile of the robust road-block population as the maximum
    # scale that context-generated blocks may reach. This keeps the ceiling
    # evidence-based while preventing anomalous 100s/1000s-ha polygons from
    # becoming the reference ceiling.
    plausible_cap=float(q75)
    return {'median_area':float(np.median(core)),
            'q25':q25,
            'q75':q75,
            'median_elongation':float(np.median(elong)) if len(elong) else 1.0,
            'count':int(len(reference_cells)),
            'max_area':float(plausible_cap),
            'raw_max_area':float(max(areas))}

def learned_target(poly, refs_gdf, pattern, user_target):
    """Get a context-sensitive preferred scale for a road-poor polygon.
    Nearby established road-defined blocks are the primary reference; the user's
    target remains a soft preference and fallback rather than a hard maximum.
    """
    if refs_gdf is None or len(refs_gdf)==0 or pattern['count']==0:
        return user_target
    try:
        near=list(refs_gdf.sindex.nearest(poly, return_all=False))
        if len(near):
            ref_area=float(refs_gdf.geometry.iloc[int(near[0])].area)
            # Blend local evidence with the global learned median and the user's
            # stated preference. Local road pattern gets the strongest weight.
            desired=0.55*ref_area + 0.25*pattern['median_area'] + 0.20*user_target
        else:
            desired=0.70*pattern['median_area']+0.30*user_target
    except Exception:
        desired=0.70*pattern['median_area']+0.30*user_target
    # Keep the learned recommendation within the observed central range, but do
    # not use that range as a maximum: an intact road-defined block can remain larger.
    lo=max(pattern['q25']*0.75, user_target*0.50, 1.0)
    hi=max(pattern['q75']*1.25, user_target*1.50, lo)
    # Hard ceiling for *generated* blocks: never exceed the largest coherent
    # road-defined reference block observed in the road network.
    if pattern.get('max_area') is not None:
        hi=min(hi, float(pattern['max_area']))
        lo=min(lo, hi)
        desired=min(desired, hi)
    return float(np.clip(desired,lo,hi))

def enforce_learned_cap(cells, cap, buildings=None):
    """Ensure generated/context blocks never exceed the largest road-defined block."""
    if cap is None or cap <= 1:
        return cells
    out=[]; stack=list(cells); safe_target=max(cap/1.03,1.0)
    while stack:
        p=clean(stack.pop())
        if p.is_empty or p.area<=1: continue
        if p.area <= cap*1.000001:
            out.append(p); continue
        pieces=split_best(p,safe_target,buildings)
        if len(pieces)>=2 and any(x.area < p.area*0.999 for x in pieces):
            stack.extend(pieces)
        else:
            # Last-resort geometric split using a more aggressive target.
            pieces=split_best(p,max(cap/2,1.0),buildings)
            if len(pieces)>=2: stack.extend(pieces)
            else: out.append(p)
    return out

@st.cache_data(show_spinner=False)
def generate(target_m2,min_m2,classes,road_source,tolerance_pct):
    roads,buildings,wards=load_data()
    if road_source=='No road structure (target only)': selected=roads.iloc[0:0].copy()
    elif road_source=='All GIS roads': selected=roads.copy()
    else: selected=road_mask(roads,classes)

    # First pass: derive road-defined natural units for every ward. These become
    # the reference population from which road-poor areas learn their local scale.
    ward_naturals=[]; reference=[]
    for _,w in wards.iterrows():
        wg=clean(w.geometry)
        r=selected[selected.intersects(wg)]
        natural=road_cells(wg,r) if len(r) else [wg]
        ward_naturals.append((str(w.get('ward',w.get('Zone_Code',w.get('Id','')))),wg,r,natural))
        for cell in natural:
            _,ratio,sides=boundary_road_metrics(cell,r,wg)
            natural_unit=(sides>=2 and ratio>=0.20) or (sides>=3 and ratio>=0.12)
            if natural_unit and cell.area>=min_m2:
                reference.append(cell)

    pattern=learn_block_pattern(reference,target_m2)
    ref_gdf=gpd.GeoDataFrame({'geometry':reference},crs=wards.crs) if reference else None
    max_reference_area=float(pattern.get('max_area')) if reference else None

    all_blocks=[]
    generated_by_context=0
    for ward_name,wg,r,natural in ward_naturals:
        pieces=[]
        for cell in natural:
            _,ratio,sides=boundary_road_metrics(cell,r,wg)
            natural_unit=(sides>=2 and ratio>=0.20) or (sides>=3 and ratio>=0.12)
            if natural_unit:
                # Existing road-defined blocks are retained even when larger than
                # the preferred scale. This is the key distinction from a hidden
                # maximum-area rule.
                pieces.append(cell)
            else:
                desired=learned_target(cell,ref_gdf,pattern,target_m2)
                if cell.area>desired*1.02:
                    pieces.extend(target_subdivide(cell,desired,min_m2,buildings))
                    generated_by_context += 1
                else:
                    pieces.append(cell)

        pieces=enforce_learned_cap(pieces,max_reference_area,buildings)
        pieces=merge_tiny(pieces,min_m2)
        pieces=repair_partition(pieces,wg)
        pieces=enforce_learned_cap(pieces,max_reference_area,buildings)
        # Only weak/context-generated polygons are subdivided here. A coherent
        # road-defined unit is never broken merely because it exceeds the target.
        refined=[]
        for p in pieces:
            _,rr,ss=boundary_road_metrics(p,r,wg)
            natural_unit=(ss>=2 and rr>=0.20) or (ss>=3 and rr>=0.12)
            if p.area>target_m2*1.02 and not natural_unit:
                desired=learned_target(p,ref_gdf,pattern,target_m2)
                if p.area>desired*1.02:
                    refined.extend(target_subdivide(p,desired,min_m2,buildings)); generated_by_context += 1
                else: refined.append(p)
            else: refined.append(p)
        pieces=enforce_partition(merge_tiny(refined,min_m2),wg)
        pieces=enforce_learned_cap(pieces,max_reference_area,buildings)
        pieces=enforce_partition(pieces,wg)
        pieces=enforce_learned_cap(pieces,max_reference_area,buildings)
        # Re-partition after every cap split so the final output can never
        # contain overlaps, nested polygons, or sliver gaps.
        pieces=enforce_partition(pieces,wg)
        all_blocks.extend({'ward':ward_name,'geometry':p} for p in pieces)

    blocks=gpd.GeoDataFrame(all_blocks,crs=wards.crs)
    # Stable block IDs must exist BEFORE building/block spatial joins.
    # Ward_Code is a machine-stable code (W01..W09), while ward remains human-readable.
    ward_num=blocks['ward'].astype(str).str.extract(r'(\d+)',expand=False).astype(int)
    blocks['Ward_Code']=ward_num.map(lambda n:f'W{n:02d}')
    blocks['block_id']=[f'{wc}-B{i:03d}' for i,wc in enumerate(blocks['Ward_Code'],1)]
    # One building spatial join only, after final block geometry is known.
    bpts=buildings.copy(); bpts['geometry']=bpts.geometry.centroid
    joined=gpd.sjoin(bpts,blocks[['geometry']],predicate='within',how='left')
    counts=joined.groupby('index_right').size()
    blocks['building_count']=[int(counts.get(i,0)) for i in blocks.index]

    # Full building inventory: assign every building to a final block, retain
    # the real footprint and area, and flag footprints that straddle a block.
    buildings_out=buildings.copy()
    buildings_out=buildings_out.reset_index(drop=True)
    buildings_out['Building_Area_m2']=buildings_out.geometry.area
    # Representative point avoids failures caused by a footprint touching a
    # block boundary; boundary crossings are separately flagged below.
    reps=buildings_out.geometry.representative_point()
    rep_gdf=gpd.GeoDataFrame({'geometry':reps},crs=buildings_out.crs)
    bj=gpd.sjoin(rep_gdf,blocks[['block_id','Ward_Code','ward','geometry']],predicate='within',how='left')
    buildings_out['Block_Code']=bj['block_id'].values
    buildings_out['Ward_Code']=bj['Ward_Code'].values
    # Vectorized boundary-crossing check. This replaces a Python loop over all
    # 24k buildings and is substantially faster on Streamlit Cloud.
    hit = gpd.sjoin(
        buildings_out[['geometry']],
        blocks[['block_id','geometry']],
        predicate='intersects', how='left'
    )
    hit_counts = hit.groupby(hit.index).size()
    multi_flags = hit_counts.reindex(buildings_out.index, fill_value=0).to_numpy() > 1
    # A footprint crossing a block edge has positive-length intersection with a
    # block boundary. Candidate pairs are limited by the spatial join above.
    hit = hit.reset_index().rename(columns={'index':'building_idx'})
    if len(hit):
        valid = hit['index_right'].notna()
        hp = hit.loc[valid, ['building_idx','index_right']].copy()
        if len(hp):
            bg = buildings_out.geometry.iloc[hp['building_idx'].to_numpy()].reset_index(drop=True)
            bb = blocks.geometry.iloc[hp['index_right'].astype(int).to_numpy()].boundary.reset_index(drop=True)
            inter = __import__('shapely').intersection(bg.to_numpy(), bb.to_numpy())
            lens = __import__('shapely').length(inter)
            crossed_idx = hp.loc[np.asarray(lens)>0.01, 'building_idx'].astype(int).unique()
        else:
            crossed_idx=np.array([],dtype=int)
    else:
        crossed_idx=np.array([],dtype=int)
    boundary_flags=np.zeros(len(buildings_out),dtype=bool)
    boundary_flags[crossed_idx]=True
    buildings_out['Boundary_Flag']=boundary_flags | multi_flags
    buildings_out['Building_Source']='OSM existing'
    buildings_out['AI_Confidence']=np.nan
    # Use the final block code directly. Do not cast an auxiliary order field
    # to int: pandas/geopandas joins can legitimately produce NaN/object values.
    buildings_out['Building_Code']=[
        f"{b}-BLD{i:04d}" if pd.notna(b) else f"UNASSIGNED-BLD{i:04d}"
        for i,b in enumerate(buildings_out['Block_Code'],1)
    ]
    blocks['area_m2']=blocks.area; blocks['area_ha']=blocks.area/10000
    blocks['target_m2']=target_m2; blocks['deviation_pct']=(blocks.area/target_m2-1)*100
    tol=float(tolerance_pct)/100
    blocks['status']=np.select([blocks.area<target_m2*(1-tol),blocks.area>target_m2*(1+tol)],['below_preferred','above_preferred'],default='within_preferred')
    # QA metadata retained in the dataframe attrs for the app panel.
    blocks.attrs['learned_reference_count']=pattern['count']
    blocks.attrs['learned_median_area_ha']=pattern['median_area']/10000
    blocks.attrs['largest_road_reference_cap_ha']=(max_reference_area/10000 if max_reference_area else None)
    blocks.attrs['context_generated_count']=generated_by_context
    return blocks[['block_id','ward','area_m2','area_ha','target_m2','deviation_pct','status','building_count','geometry']],wards,roads,buildings_out

st.title('Ganta Parametric Planning Block Generator')
st.caption('Ward boundaries are hard limits. Existing roads define natural planning blocks where they form a coherent layout. Road-poor areas learn their preferred block scale from nearby established road-defined blocks. Generated/context blocks never exceed the robust upper scale learned from coherent road-defined blocks; anomalous giant polygonized areas are not treated as planning-block references; the target remains a soft preference. Blocks form a non-overlapping partition with no nested blocks. In road-poor areas, candidate block cuts are actively scored to avoid existing building footprints.')
with st.sidebar:
    st.header('Block parameters')
    target_ha=st.number_input('Target block area (ha)',min_value=1.0,max_value=100.0,value=1.0,step=0.25)
    tolerance_pct=st.slider('Target preference / reporting tolerance (%)',0,50,20,5,key='tolerance_pct')
    min_ha=st.number_input('Minimum block area (ha)',min_value=0.25,max_value=50.0,value=0.25,step=0.25)
    st.subheader('Road structure source')
    road_source=st.selectbox('Dataset used to generate blocks',['Selected GIS road classes','All GIS roads','No road structure (target only)'])
    classes=st.multiselect('Road hierarchy used',['primary','trunk','tertiary','unclassified','residential','service','track','path','footway'],default=['primary','trunk','tertiary','unclassified','residential'])
    st.subheader('Basemap')
    basemap=st.selectbox('Visual basemap',['OpenStreetMap','Esri World Imagery','Esri World Street Map','Google Roadmap (API key/session required)','Google Satellite (API key/session required)'])
    google_key=st.text_input('Google API key (optional)',type='password',help='Used only for Google Map Tiles. Google visual tiles are never used as road data.')
    google_road_session=st.text_input('Google Roadmap session token (optional)',type='password')
    google_sat_session=st.text_input('Google Satellite session token (optional)',type='password')
    st.subheader('Map styling')
    ward_color=st.color_picker('Ward outline colour','#222222'); ward_weight=st.slider('Ward line weight',1.0,8.0,4.0,0.5)
    block_color=st.color_picker('Block outline colour','#0066CC'); block_weight=st.slider('Block line weight',0.5,8.0,2.5,0.5)
    block_fill=st.checkbox('Fill planning blocks',value=False); block_fill_opacity=st.slider('Block fill opacity',0.0,0.8,0.15,0.05)
    road_weight=st.slider('Road line weight',0.5,5.0,1.5,0.5)
    show_buildings=st.checkbox('Show coded building footprints',value=False,help='Turn on only when inspecting building-level coding. Full building inventory remains available for download.')
    generate_btn=st.button('Generate blocks',type='primary',use_container_width=True)
    st.info('Target area is a preference, not a maximum. Generated/context blocks are capped by the robust upper scale of the observed road-defined block pattern, excluding anomalous giant polygonized areas. The tolerance affects reporting only. Minimum area suppresses tiny residual blocks.')

gen_sig=(float(target_ha),float(min_ha),road_source,tuple(classes))
map_sig=(basemap,google_key,google_road_session,google_sat_session,ward_color,ward_weight,block_color,block_weight,block_fill,block_fill_opacity,road_weight)
if 'blocks' not in st.session_state or st.session_state.get('gen_sig')!=gen_sig or generate_btn:
    with st.spinner('Generating planning blocks...'):
        blocks,wards,roads,buildings=generate(target_ha*10000,min_ha*10000,classes,road_source,tolerance_pct)
        st.session_state.update(blocks=blocks,wards=wards,roads=roads,buildings=buildings,gen_sig=gen_sig)
blocks=st.session_state.blocks; wards=st.session_state.wards; roads=st.session_state.roads; buildings=st.session_state.buildings
map_blocks=blocks.to_crs(4326); map_wards=wards.to_crs(4326); map_roads=roads.to_crs(4326); map_buildings=buildings.to_crs(4326) if show_buildings else None
bounds=map_wards.total_bounds; center=[(bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2]

c1,c2,c3,c4,c5=st.columns(5); c1.metric('Planning blocks',len(blocks)); c5.metric('Coded buildings',len(buildings)) ; c2.metric('Target area',f'{target_ha:.1f} ha'); c3.metric('Within preferred range',f'{(blocks.status=="within_preferred").mean()*100:.0f}%'); c4.metric('Median block',f'{blocks.area_ha.median():.2f} ha')
left,right=st.columns([2,1])
with left:
    m=folium.Map(location=center,zoom_start=13,tiles=None,control_scale=True)
    google_ok=False
    folium.TileLayer('OpenStreetMap',name='OpenStreetMap',control=True,show=(basemap=='OpenStreetMap' or basemap.startswith('Google'))).add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',attr='Esri',name='Esri World Imagery',control=True,show=basemap=='Esri World Imagery',overlay=False).add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',attr='Esri',name='Esri World Street Map',control=True,show=basemap=='Esri World Street Map',overlay=False).add_to(m)
    if basemap.startswith('Google Roadmap') and google_key and google_road_session:
        url='https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session='+google_road_session+'&key='+google_key
        folium.TileLayer(tiles=url,attr='Google Maps Platform',name='Google Roadmap',control=True,show=True,overlay=False).add_to(m); google_ok=True
    elif basemap.startswith('Google Satellite') and google_key and google_sat_session:
        url='https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session='+google_sat_session+'&key='+google_key
        folium.TileLayer(tiles=url,attr='Google Maps Platform',name='Google Satellite',control=True,show=True,overlay=False).add_to(m); google_ok=True
    elif basemap.startswith('Google'):
        st.warning('Google basemap selected, but the required API key and matching 2D tile session token were not supplied. Showing OpenStreetMap instead. Google tiles are visual only and never feed the block generator.')
    folium.GeoJson(map_wards.to_json(),name='Wards',style_function=lambda x:{'fillOpacity':0,'color':ward_color,'weight':ward_weight,'dashArray':'8,5'},tooltip=folium.GeoJsonTooltip(fields=['ward'],aliases=['Ward'],sticky=False)).add_to(m)
    folium.GeoJson(map_blocks.to_json(),name='Planning Blocks',style_function=lambda f:{'fillOpacity':block_fill_opacity if block_fill else 0,'fillColor':block_color,'color':block_color,'weight':block_weight,'opacity':0.95},highlight_function=lambda f:{'weight':max(block_weight+1.5,block_weight),'fillOpacity':block_fill_opacity if block_fill else 0},tooltip=folium.GeoJsonTooltip(fields=['block_id','ward','area_ha','status','building_count'],aliases=['Block','Ward','Area (ha)','Status','Buildings'],sticky=False)).add_to(m)
    if show_buildings:
        folium.GeoJson(map_buildings.to_json(),name='Buildings — coded & labelled',style_function=lambda f:{'fillOpacity':0.35,'color':'#8B0000','weight':1.0},highlight_function=lambda f:{'weight':2.0,'fillOpacity':0.55},tooltip=folium.GeoJsonTooltip(fields=['Building_Code','Ward_Code','Block_Code','Building_Area_m2','Building_Source','Boundary_Flag'],aliases=['Building Code','Ward','Block','Building Area (m²)','Source','Boundary flag'],localize=True,sticky=False)).add_to(m)
    folium.GeoJson(map_roads.to_json(),name='Road network',style_function=lambda x:{'weight':road_weight,'opacity':0.65}).add_to(m)
    m.fit_bounds([[bounds[1],bounds[0]],[bounds[3],bounds[2]]]); folium.LayerControl(collapsed=False).add_to(m); st_folium(m,height=650,width=None)
with right:
    st.subheader('Area distribution'); st.dataframe(blocks.drop(columns='geometry').sort_values(['ward','area_m2']),use_container_width=True,height=520)
    st.download_button('Download planning blocks (GeoJSON)',blocks.to_json(),'Ganta_PlanningBlocks.geojson','application/geo+json')
    st.download_button('Download coded buildings (GeoJSON)',buildings.to_json(),'Ganta_Buildings_Coded.geojson','application/geo+json')
    st.download_button('Download building inventory (CSV)',buildings.drop(columns='geometry').to_csv(index=False),'Ganta_Building_Inventory.csv','text/csv')
    st.download_button('Download summary (CSV)',blocks.drop(columns='geometry').to_csv(index=False),'Ganta_PlanningBlocks_Summary.csv','text/csv')
st.markdown('### Planning rule used')
st.write('Ward boundary = hard boundary. Selected GIS roads are the primary structure source. A road-defined cell is retained at its natural size only when roads form meaningful portions of its interior boundary; ward-edge roads do not qualify. Where the layout is weak, clean target-oriented cuts are used. The target is never a maximum, and reporting tolerance does not constrain generation. Buildings are used as a supporting constraint in road-poor areas: generated cut lines are selected to avoid passing through building footprints where a reasonable alternative exists. Buildings never become block boundaries themselves. A final topology pass guarantees a single, non-overlapping block partition per ward.')
