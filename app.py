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

@st.cache_data
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
    # Only roads forming the candidate boundary count. Ward boundary is explicitly excluded.
    b=cell.boundary
    try:
        hits=[]
        for g in roads.geometry:
            if g is None or g.is_empty: continue
            inter=b.intersection(g)
            if not inter.is_empty and inter.length>5: hits.append(float(inter.length))
        total=sum(hits); ratio=total/max(b.length,1.0)
        meaningful=sum(1 for h in hits if h>=15)
        return total,ratio,meaningful
    except Exception:return 0.0,0.0,0

def split_best(poly,target):
    poly=clean(poly)
    if poly.is_empty or poly.area<=target*1.02:return [poly]
    if poly.geom_type=='MultiPolygon':
        out=[]
        for p in poly.geoms: out.extend(split_best(p,target))
        return out
    minx,miny,maxx,maxy=poly.bounds; w=maxx-minx; h=maxy-miny
    # Try both orientations and several positions; score partition by closeness of resulting pieces to target.
    cand=[]
    for use_x in ([True,False] if abs(w-h)/max(w,h,1)>0.12 else [True,False]):
        lo,hi=(minx,maxx) if use_x else (miny,maxy); span=hi-lo
        if span<=0:continue
        for frac in np.linspace(0.15,0.85,15):
            v=lo+span*float(frac); pad=max(w,h)*3+10
            ln=LineString([(v,miny-pad),(v,maxy+pad)]) if use_x else LineString([(minx-pad,v),(maxx+pad,v)])
            try: pieces=[clean(x) for x in split(poly,ln).geoms if x.area>1]
            except Exception:continue
            if len(pieces)<2:continue
            # Recursive partition is done outside; first cut should leave no very tiny piece.
            areas=np.array([p.area for p in pieces]); small=(areas<0.25*target).sum()
            score=abs(np.median(areas)-target)/max(target,1)+0.8*small+0.15*abs(areas.max()-target)/max(target,1)
            cand.append((score,pieces))
    if not cand:return [poly]
    return min(cand,key=lambda x:x[0])[1]

def target_subdivide(poly,target,min_area):
    out=[]; stack=[poly]; guard=0
    while stack and guard<10000:
        guard+=1; p=clean(stack.pop())
        if p.is_empty or p.area<=1:continue
        if p.area<=target*1.02:
            out.append(p);continue
        pieces=split_best(p,target)
        if len(pieces)<2:out.append(p)
        else:stack.extend(pieces)
    return out

def merge_tiny(cells,min_area):
    cells=[clean(c) for c in cells if c is not None and not c.is_empty and c.area>1]
    while True:
        idx=[i for i,c in enumerate(cells) if c.area<min_area]
        if not idx or len(cells)<=1:break
        i=min(idx,key=lambda k:cells[k].area); best=None; bestscore=-1
        for j,c in enumerate(cells):
            if j==i:continue
            shared=cells[i].boundary.intersection(c.boundary).length
            if shared>bestscore:bestscore=shared;best=j
        if best is None or bestscore<=0.01:break
        merged=clean(cells[i].union(cells[best]))
        cells=[c for k,c in enumerate(cells) if k not in (i,best)]+[merged]
    return cells

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

def generate(target_m2,min_m2,classes,road_source):
    roads,buildings,wards=load_data()
    if road_source=='No road structure (target only)': selected=roads.iloc[0:0].copy()
    elif road_source=='All GIS roads': selected=roads.copy()
    else:selected=road_mask(roads,classes)
    all_blocks=[]
    for _,w in wards.iterrows():
        wg=clean(w.geometry)
        r=selected[selected.intersects(wg)]
        natural=road_cells(wg,r) if len(r) else [wg]
        pieces=[]
        for cell in natural:
            road_len,ratio,sides=boundary_road_metrics(cell,r,wg)
            # A cell is genuinely road-defined only if roads form substantial boundary structure.
            natural_unit=(sides>=2 and ratio>=0.20) or (sides>=3 and ratio>=0.12)
            if natural_unit:
                pieces.append(cell)
            else:
                pieces.extend(target_subdivide(cell,target_m2,min_m2))
        pieces=merge_tiny(pieces,min_m2)
        pieces=repair_partition(pieces,wg)
        # Second pass: only weak, very large pieces are target-subdivided; road-defined large units stay intact.
        refined=[]
        for p in pieces:
            rl,rr,ss=boundary_road_metrics(p,r,wg)
            natural_unit=(ss>=2 and rr>=0.20) or (ss>=3 and rr>=0.12)
            if (not natural_unit) and p.area>target_m2*1.02:
                refined.extend(target_subdivide(p,target_m2,min_m2))
            else:refined.append(p)
        pieces=repair_partition(merge_tiny(refined,min_m2),wg)
        ward_name=str(w.get('ward',w.get('Zone_Code',w.get('Id',''))))
        for p in pieces:all_blocks.append({'ward':ward_name,'geometry':p})
    blocks=gpd.GeoDataFrame(all_blocks,crs=wards.crs)
    bpts=buildings.copy(); bpts['geometry']=bpts.geometry.centroid
    joined=gpd.sjoin(bpts,blocks[['geometry']],predicate='within',how='left')
    counts=joined.groupby('index_right').size()
    blocks['building_count']=[int(counts.get(i,0)) for i in blocks.index]
    blocks['area_m2']=blocks.area; blocks['area_ha']=blocks.area/10000
    blocks['target_m2']=target_m2; blocks['deviation_pct']=(blocks.area/target_m2-1)*100
    tol=st.session_state.get('tolerance_pct',20)/100
    blocks['status']=np.select([blocks.area<target_m2*(1-tol),blocks.area>target_m2*(1+tol)],['below_preferred','above_preferred'],default='within_preferred')
    blocks['block_id']=[f'{w}-B{i:03d}' for i,w in enumerate(blocks.ward,1)]
    return blocks[['block_id','ward','area_m2','area_ha','target_m2','deviation_pct','status','building_count','geometry']],wards,roads,buildings

st.title('Ganta Parametric Planning Block Generator')
st.caption('Ward boundaries are hard limits. Roads define natural planning blocks where they form a coherent layout; the target guides subdivision only where the road structure is weak.')
with st.sidebar:
    st.header('Block parameters')
    target_ha=st.number_input('Target block area (ha)',min_value=1.0,max_value=100.0,value=3.0,step=0.25)
    tolerance_pct=st.slider('Target preference / reporting tolerance (%)',0,50,20,5,key='tolerance_pct')
    min_ha=st.number_input('Minimum block area (ha)',min_value=0.25,max_value=50.0,value=0.25,step=0.25)
    st.subheader('Road structure source')
    road_source=st.selectbox('Dataset used to generate blocks',['Selected GIS road classes','All GIS roads','No road structure (target only)'])
    classes=st.multiselect('Road hierarchy used',['primary','trunk','tertiary','unclassified','residential','service','track','path','footway'],default=['primary','trunk','tertiary','unclassified','residential','service','track','path','footway'])
    st.subheader('Basemap')
    basemap=st.selectbox('Visual basemap',['Esri World Street Map','OpenStreetMap','Esri World Imagery','Google Roadmap (API key/session required)','Google Satellite (API key/session required)'])
    google_key=st.text_input('Google API key (optional)',type='password',help='Used only for Google Map Tiles. Google visual tiles are never used as road data.')
    google_road_session=st.text_input('Google Roadmap session token (optional)',type='password')
    google_sat_session=st.text_input('Google Satellite session token (optional)',type='password')
    st.subheader('Map styling')
    ward_color=st.color_picker('Ward outline colour','#222222'); ward_weight=st.slider('Ward line weight',1.0,8.0,4.0,0.5)
    block_color=st.color_picker('Block outline colour','#0066CC'); block_weight=st.slider('Block line weight',0.5,8.0,2.5,0.5)
    block_fill=st.checkbox('Fill planning blocks',value=False); block_fill_opacity=st.slider('Block fill opacity',0.0,0.8,0.15,0.05)
    road_weight=st.slider('Road line weight',0.5,5.0,1.5,0.5)
    generate_btn=st.button('Generate blocks',type='primary',use_container_width=True)
    st.info('Target area is a preference, not a maximum. No hidden target-derived ceiling is used. The tolerance affects reporting only. Minimum area is used to suppress tiny residual blocks.')

sig=(float(target_ha),int(tolerance_pct),float(min_ha),road_source,tuple(classes),basemap,google_key,google_road_session,google_sat_session)
if 'blocks' not in st.session_state or st.session_state.get('sig')!=sig or generate_btn:
    with st.spinner('Generating planning blocks...'):
        blocks,wards,roads,buildings=generate(target_ha*10000,min_ha*10000,classes,road_source)
        st.session_state.update(blocks=blocks,wards=wards,roads=roads,buildings=buildings,sig=sig)
blocks=st.session_state.blocks; wards=st.session_state.wards; roads=st.session_state.roads
map_blocks=blocks.to_crs(4326); map_wards=wards.to_crs(4326); map_roads=roads.to_crs(4326)
bounds=map_wards.total_bounds; center=[(bounds[1]+bounds[3])/2,(bounds[0]+bounds[2])/2]

c1,c2,c3,c4=st.columns(4); c1.metric('Planning blocks',len(blocks)); c2.metric('Target area',f'{target_ha:.1f} ha'); c3.metric('Within preferred range',f'{(blocks.status=="within_preferred").mean()*100:.0f}%'); c4.metric('Median block',f'{blocks.area_ha.median():.2f} ha')
left,right=st.columns([2,1])
with left:
    m=folium.Map(location=center,zoom_start=13,tiles=None,control_scale=True)
    folium.TileLayer('OpenStreetMap',name='OpenStreetMap',control=True,show=(basemap=='OpenStreetMap' or (basemap.startswith('Google') and not google_ok))).add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',attr='Esri',name='Esri World Imagery',control=True,show=basemap=='Esri World Imagery',overlay=False).add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',attr='Esri',name='Esri World Street Map',control=True,show=basemap=='Esri World Street Map',overlay=False).add_to(m)
    google_ok=False
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
    folium.GeoJson(map_roads.to_json(),name='Road network',style_function=lambda x:{'weight':road_weight,'opacity':0.65}).add_to(m)
    m.fit_bounds([[bounds[1],bounds[0]],[bounds[3],bounds[2]]]); folium.LayerControl(collapsed=False).add_to(m); st_folium(m,height=650,width=None)
with right:
    st.subheader('Area distribution'); st.dataframe(blocks.drop(columns='geometry').sort_values(['ward','area_m2']),use_container_width=True,height=520)
    st.download_button('Download planning blocks (GeoJSON)',blocks.to_json(),'Ganta_PlanningBlocks.geojson','application/geo+json')
    st.download_button('Download summary (CSV)',blocks.drop(columns='geometry').to_csv(index=False),'Ganta_PlanningBlocks_Summary.csv','text/csv')
st.markdown('### Planning rule used')
st.write('Ward boundary = hard boundary. Selected GIS roads are the primary structure source. A road-defined cell is retained at its natural size when roads form meaningful portions of its boundary. Where the layout is weak, clean target-oriented cuts are used. The target is never a maximum, and reporting tolerance does not constrain generation. Buildings are used only for validation/reporting, not as block boundaries.')
