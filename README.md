# Ganta Parametric Planning Block Generator — v22

This version fixes a Streamlit Cloud runtime failure caused by a spatial join returning 24,681 rows for 24,680 buildings. Building-to-block assignment now de-duplicates the join by source building index and reindexes to the full building inventory before assigning codes.

It retains the v21 bounded, building-aware generator and building-avoidance logic.
