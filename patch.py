from pathlib import Path
p=Path('/mnt/data/ganta_work/v16_work/app.py')
s=p.read_text()
# replace split_best definition through target_subdivide start
start=s.index('def split_best(')
end=s.index('def merge_tiny(', start)
new=r'''def _line_building_penalty(line, buildings):
    """Measure how much a proposed block cut passes through buildings.
    The score is deliberately strong: in road-poor areas we prefer a slightly
    less area-balanced cut if it avoids existing building footprints."""
    if buildings is None or len(buildings) == 0:
        return 0.0, 0
    try:
        hits = buildings.sindex.query(line, predicate='intersects')
    except Exception:
        hits = []
    if len(hits) == 0:
        return 0.0, 0
    total = 0.0
    count = 0
    for idx in hits:
        try:
            inter = line.intersection(buildings.geometry.iloc[int(idx)])
            if not inter.is_empty:
                total += float(inter.length)
                count += 1
        except Exception:
            pass
    return total, count

def split_best(poly, target, buildings=None):
    """Fast deterministic split that actively avoids cutting through buildings.
    Candidate cuts are tested against the building spatial index. A no-building
    cut wins whenever its area balance is reasonably acceptable."""
    poly=clean(poly)
    if poly.is_empty or poly.area<=target*1.02:return [poly]
    if poly.geom_type=='MultiPolygon':
        out=[]
        for p in poly.geoms: out.extend(split_best(p,target,buildings))
        return out
    minx,miny,maxx,maxy=poly.bounds; w=maxx-minx; h=maxy-miny
    # Test both axes where possible; this gives the algorithm a chance to move
    # around building clusters instead of always slicing on one axis.
    axes=[]
    if w>=h: axes=['x','y']
    else: axes=['y','x']
    candidates=[]
    for axis in axes:
        lo,hi=(minx,maxx) if axis=='x' else (miny,maxy)
        span=hi-lo
        if span<=0: continue
        for frac in (0.35,0.45,0.50,0.55,0.65):
            v=lo+span*frac; pad=max(w,h)*2+10
            ln=LineString([(v,miny-pad),(v,maxy+pad)]) if axis=='x' else LineString([(minx-pad,v),(maxx+pad,v)])
            try: pieces=[clean(x) for x in split(poly,ln).geoms if x.area>1]
            except Exception: continue
            if len(pieces)<2: continue
            areas=np.array([p.area for p in pieces])
            small=(areas<0.25*target).sum()
            balance=abs(np.median(areas)-target)/max(target,1)
            oversize=max(areas.max()/max(target,1)-1,0)
            cut_len,cut_count=_line_building_penalty(ln,buildings)
            # A cut crossing even one building is heavily penalized. Zero-crossing
            # candidates therefore dominate unless they create pathological pieces.
            building_penalty=25.0*cut_count + 80.0*cut_len/max(np.sqrt(poly.area),1.0)
            score=balance + 1.5*small + 0.10*oversize + building_penalty
            candidates.append((score,cut_count,cut_len,pieces))
    if not candidates:return [poly]
    # Prefer cuts that do not cross buildings; among those use geometry score.
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

'''
s=s[:start]+new+s[end:]
# calls add buildings
s=s.replace('target_subdivide(cell,desired,min_m2)', 'target_subdivide(cell,desired,min_m2,buildings)')
s=s.replace('target_subdivide(p,desired,min_m2)', 'target_subdivide(p,desired,min_m2,buildings)')
s=s.replace('split_best(p,safe_target)', 'split_best(p,safe_target,buildings=None)')
s=s.replace('split_best(p,max(cap/2,1.0))', 'split_best(p,max(cap/2,1.0),buildings=None)')
# Fix enforce_learned_cap: it doesn't have buildings, leave it geometry-only.
# Add building coding after block generation/join section
old="""    joined=gpd.sjoin(bpts,blocks[['geometry']],predicate='within',how='left')\n    counts=joined.groupby('index_right').size()\n    blocks['building_count']=[int(counts.get(i,0)) for i in blocks.index]\n"""
new2="""    joined=gpd.sjoin(bpts,blocks[['geometry']],predicate='within',how='left')\n    counts=joined.groupby('index_right').size()\n    blocks['building_count']=[int(counts.get(i,0)) for i in blocks.index]\n\n    # Full building inventory: assign every building to a final block, retain\n    # the real footprint and area, and flag footprints that straddle a block.\n    buildings_out=buildings.copy()\n    buildings_out=buildings_out.reset_index(drop=True)\n    buildings_out['Building_Area_m2']=buildings_out.geometry.area\n    # Representative point avoids failures caused by a footprint touching a\n    # block boundary; boundary crossings are separately flagged below.\n    reps=buildings_out.geometry.representative_point()\n    rep_gdf=gpd.GeoDataFrame({'geometry':reps},crs=buildings_out.crs)\n    bj=gpd.sjoin(rep_gdf,blocks[['block_id','ward','geometry']],predicate='within',how='left')\n    buildings_out['Block_Code']=bj['block_id'].values\n    buildings_out['Ward_Code']=bj['ward'].values\n    # A building is flagged if its footprint intersects more than one block or\n    # if its boundary is actually crossed by a block edge. We never clip it.\n    b_sidx=blocks.sindex\n    boundary_flags=[]\n    multi_flags=[]\n    for geom in buildings_out.geometry:\n        try: hits=list(b_sidx.query(geom,predicate='intersects'))\n        except Exception: hits=[]\n        multi_flags.append(len(set(map(int,hits)))>1)\n        crossed=False\n        try:\n            for hi in hits:\n                inter=geom.intersection(blocks.geometry.iloc[int(hi)].boundary)\n                if not inter.is_empty and inter.length>0.01:\n                    crossed=True; break\n        except Exception: pass\n        boundary_flags.append(crossed)\n    buildings_out['Boundary_Flag']=[bool(a or b) for a,b in zip(boundary_flags,multi_flags)]\n    buildings_out['Building_Source']='OSM existing'\n    buildings_out['AI_Confidence']=np.nan\n    buildings_out['Building_Code']=[\n        f"{w}-B{int(str(b).split('-B')[-1]):03d}-BLD{i:04d}" if pd.notna(w) and pd.notna(b) else f"UNASSIGNED-BLD{i:04d}"\n        for i,(w,b) in enumerate(zip(buildings_out['Ward_Code'],buildings_out['Block_Code']),1)\n    ]\n"""
s=s.replace(old,new2)
# return buildings_out
s=s.replace("return blocks[['block_id','ward','area_m2','area_ha','target_m2','deviation_pct','status','building_count','geometry']],wards,roads,buildings", "return blocks[['block_id','ward','area_m2','area_ha','target_m2','deviation_pct','status','building_count','geometry']],wards,roads,buildings_out")
# session assignment and map vars
s=s.replace("blocks=st.session_state.blocks; wards=st.session_state.wards; roads=st.session_state.roads", "blocks=st.session_state.blocks; wards=st.session_state.wards; roads=st.session_state.roads; buildings=st.session_state.buildings")
s=s.replace("map_blocks=blocks.to_crs(4326); map_wards=wards.to_crs(4326); map_roads=roads.to_crs(4326)", "map_blocks=blocks.to_crs(4326); map_wards=wards.to_crs(4326); map_roads=roads.to_crs(4326); map_buildings=buildings.to_crs(4326)")
# insert building layer after blocks layer
needle="""    folium.GeoJson(map_blocks.to_json(),name='Planning Blocks',style_function=lambda f:{'fillOpacity':block_fill_opacity if block_fill else 0,'fillColor':block_color,'color':block_color,'weight':block_weight,'opacity':0.95},highlight_function=lambda f:{'weight':max(block_weight+1.5,block_weight),'fillOpacity':block_fill_opacity if block_fill else 0},tooltip=folium.GeoJsonTooltip(fields=['block_id','ward','area_ha','status','building_count'],aliases=['Block','Ward','Area (ha)','Status','Buildings'],sticky=False)).add_to(m)\n"""
insert=needle+"""    folium.GeoJson(map_buildings.to_json(),name='Buildings — coded & labelled',style_function=lambda f:{'fillOpacity':0.35,'color':'#8B0000','weight':1.0},highlight_function=lambda f:{'weight':2.0,'fillOpacity':0.55},tooltip=folium.GeoJsonTooltip(fields=['Building_Code','Ward_Code','Block_Code','Building_Area_m2','Building_Source','Boundary_Flag'],aliases=['Building Code','Ward','Block','Building Area (m²)','Source','Boundary flag'],localize=True,sticky=False)).add_to(m)\n"""
s=s.replace(needle,insert)
# metrics add building count
s=s.replace("c1,c2,c3,c4=st.columns(4); c1.metric('Planning blocks',len(blocks));", "c1,c2,c3,c4,c5=st.columns(5); c1.metric('Planning blocks',len(blocks)); c5.metric('Coded buildings',len(buildings)) ;")
# replace columns because 5 columns still c2 etc okay.
# Add building downloads before summary download
needle2="""    st.download_button('Download planning blocks (GeoJSON)',blocks.to_json(),'Ganta_PlanningBlocks.geojson','application/geo+json')\n    st.download_button('Download summary (CSV)',blocks.drop(columns='geometry').to_csv(index=False),'Ganta_PlanningBlocks_Summary.csv','text/csv')\n"""
rep2="""    st.download_button('Download planning blocks (GeoJSON)',blocks.to_json(),'Ganta_PlanningBlocks.geojson','application/geo+json')\n    st.download_button('Download coded buildings (GeoJSON)',buildings.to_json(),'Ganta_Buildings_Coded.geojson','application/geo+json')\n    st.download_button('Download building inventory (CSV)',buildings.drop(columns='geometry').to_csv(index=False),'Ganta_Building_Inventory.csv','text/csv')\n    st.download_button('Download summary (CSV)',blocks.drop(columns='geometry').to_csv(index=False),'Ganta_PlanningBlocks_Summary.csv','text/csv')\n"""
s=s.replace(needle2,rep2)
# change caption/rule text
s=s.replace("Blocks form a non-overlapping partition with no nested blocks.')", "Blocks form a non-overlapping partition with no nested blocks. In road-poor areas, candidate block cuts are actively scored to avoid existing building footprints.')")
s=s.replace("Buildings are used only for validation/reporting, not as block boundaries.", "Buildings are used as a supporting constraint in road-poor areas: generated cut lines are selected to avoid passing through building footprints where a reasonable alternative exists. Buildings never become block boundaries themselves.")
p.write_text(s)
