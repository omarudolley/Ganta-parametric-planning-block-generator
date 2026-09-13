# Ganta Parametric Planning Block Generator

Run with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What is fixed
- Ward boundaries are hard limits; blocks cannot cross wards.
- Blocks are a true partition: no overlapping or nested blocks.
- Tiny residuals are merged into adjacent blocks using the minimum-area setting.
- No hidden target-derived maximum is used.
- Roads determine natural block size where they form meaningful block boundaries.
- Weak/no-road areas are subdivided toward the target area.
- Buildings are not used as block boundaries.
- Block and ward outlines have independent colour/line-weight controls.
- Block fill is optional and off by default.
- Hovering a block reports Block, Ward, Area, Status, and Buildings.
- Basemap and road-structure source are independent controls.

## Basemap
OpenStreetMap and Esri basemaps work directly. Google Roadmap/Satellite options are included as API-key/session-token options. Google visual tiles are never used to generate planning blocks.

## Road structure source
- Selected GIS road classes
- All GIS roads
- No road structure (target-only)

The included `Ganta_BaseData.gpkg` remains the actual road/building source.
