# Ganta Parametric Planning Block Generator — OSM + coded buildings

This version returns to the previous OpenStreetMap-based Ganta dataset and adds a complete building inventory/annotation workflow.

## What changed

- Uses the existing `Ganta_BaseData.gpkg` OSM road and building layers.
- Keeps ward boundaries as hard limits.
- Generates non-overlapping planning blocks using the existing road/context logic.
- Target area remains a soft preference, not a hard maximum.
- Prevents nested/overlapping planning blocks through the final topology pass.
- Retains every source building polygon; buildings are **not clipped** to blocks.
- Assigns every building to the final planning block using its centroid.
- Creates stable codes:
  - Ward: `W01` … `W09`
  - Planning block: `W01-B001`
  - Building: `W01-B001-BLD0001`
- Calculates `Building_Area_m2` from the source polygon.
- Adds `Building_Source = OSM existing`.
- Adds `AI_Confidence` as a reserved field for later AI verification.
- Adds `Boundary_Flag` when a building footprint crosses its assigned block boundary.
- Adds a coded-building map layer with hover information for building code, ward, block, area, source and boundary flag.
- Provides downloads for coded building GeoJSON/CSV and planning-block GeoJSON/CSV.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The packaged app contains the source Ganta wards shapefile and OSM-based `Ganta_BaseData.gpkg`.

## Initial generated dataset

The package also includes a 1 ha-preference run using the selected OSM road classes. The generated GeoPackage contains `wards`, `roads`, `planning_blocks`, and `buildings_coded` layers.
