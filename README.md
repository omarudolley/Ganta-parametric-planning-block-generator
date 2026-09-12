# Ganta Parametric Planning Block Generator

A Streamlit web app for generating road-informed planning blocks inside the 9 Ganta wards.

## What it does

- Ward boundaries are hard limits: no block crosses a ward.
- Selected roads form the primary subdivision framework.
- Where the road network leaves very large areas, the app introduces clean subdivision lines so block sizes remain relatively similar to the target.
- Tiny fragments are merged into adjacent blocks where possible.
- Blocks are never drawn as nested/filled polygons on the map: the map uses outline-only block symbology.
- Ward outlines and planning-block outlines use different colours/styles.
- Hovering a block displays its Block ID and Ward, plus area, status and building count.
- Hovering a ward displays the Ward label.
- Buildings are used only as a validation indicator, not as a block boundary.
- Export planning blocks as GeoJSON and the summary table as CSV.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Planning interpretation

The target area (for example 20 ha / 200,000 m²) is an optimization target rather than an exact legal parcel size. The generator prioritizes ward containment, a coherent road framework, relatively even block sizes, and avoidance of tiny fragments. Artificial subdivision lines are introduced only where the road network cannot reasonably produce the target-sized blocks on its own.
