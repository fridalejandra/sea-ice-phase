# Transect diagnostics (Bootstrap 25 km)

## Purpose
Define fixed transect geometries per sector and extract reproducible distance-based diagnostics
(along-transect distance and seasonal distance-to-edge) for selecting representative pixels.

## Data / grid
- Product: Bootstrap / NSIDC polar stereographic S25km
- Grid spacing: 25 km (used only for diagnostic distances)

## Distance definitions
### 1) Along-transect distance (s_km)
Computed from the ordered transect pixel list. Each move to an adjacent grid cell contributes:
- 25 km for 4-neighbor steps
- 25*sqrt(2) km for diagonal steps (8-neighbor)
Cumulative sum from the seed pixel yields s_km.

### 2) Distance-to-edge (dist_to_edge_km)
Extracted from seasonal distance-to-edge fields:
`results/diagnostics/edge_distance/bootstrap_ws/seasonal/dist_to_edge_km_<TAG>.nc`
(where TAG ∈ {GROWTH_MAMJ, WINTER_JASO, DECAY_NDJF}).
Distance-to-edge is computed as Euclidean distance (in grid cells) to the nearest climatological
ice-edge pixel, multiplied by 25 km. Defined only inside the climatological ice mask (NaN elsewhere).

## Automatic point selection (planned)
For each sector and season TAG, compute sector-wide quantiles of dist_to_edge_km (within the sector,
optionally within the seasonal ice zone). For each transect, select pixels whose dist_to_edge_km
best match target quantiles:
- edge-like: p10
- mid-pack: p50
- interior: p90
Selections are restricted to pixels on the transect and inside the seasonal ice zone (if applied).
