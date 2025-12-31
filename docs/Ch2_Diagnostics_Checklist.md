# Chapter 2 Diagnostics Checklist

> **Goal:** empirically characterize pixel-scale SIC time-series behavior across spatial regimes *before* phase detection, and diagnose where and why seasonal transitions are ambiguous.

---

## D0. Data scope & versioning
- [ ] Confirm SIC products:
  - [ ] Bootstrap NSIDC 25 km (1979–present)
  - [ ] AMSR-E/AMSR2 12.5 km (2011–present; AMSR-E gap noted)
- [ ] Standardize calendar (365-day, no Feb 29)
- [ ] Define advance and retreat diagnostic windows
  - [ ] Advance: Mar–Jun (diagnostic)
  - [ ] Retreat: Oct–Jan (diagnostic)

---

## D1. Climatological ice-edge and distance fields (Bootstrap)
- [ ] Compute monthly or seasonal climatological **edge probability**:
  - `p_ice = mean(SIC >= 0.15)`
- [ ] Define climatological ice mask:
  - `ice_clim = p_ice >= 0.5`
- [ ] Extract climatological ice-edge boundary
- [ ] Compute **distance-to-edge field** (inside ice only)
  - Convert gridcells → km (×25 km)
- [ ] Save outputs:
  - `ice_edge_prob.nc`
  - `ice_mask_clim.nc`
  - `dist_to_edge_km.nc`

---

## D2. Transect geometry
- [ ] Define transect generation rule
  - coastal → offshore
  - along-edge (or secondary cross-shelf)
- [ ] Generate **2 transects per sector**
  - Weddell Sea
  - Amundsen–Bellingshausen
  - East Antarctica
  - Ross Sea
  - King Haakon
- [ ] Save transects:
  - `transects.geojson` or equivalent
  - include seed, direction, length metadata

---

## D3. Point selection (distance-based)
- [ ] For each transect, select 3 points:
  - **Edge:** 0–100 km inside climatological edge
  - **Mid:** 100–300 km
  - **Interior:** >300 km
- [ ] Save selection table:
  - `diagnostic_points.csv`
  - Columns: sector, transect_id, role, i/j, lat/lon, dist_km

---

## D4. Time-series diagnostics (Bootstrap)
For each selected point:
- [ ] Extract daily SIC time series
- [ ] Compute daily ΔSIC
- [ ] Compute SIC **rank/percentile** within seasonal window
- [ ] Compute metrics:
  - σ(SIC)
  - σ(ΔSIC)
  - impulse count (|ΔSIC| > threshold)
  - threshold crossing count (15% or band)
  - rank volatility
- [ ] Save metrics table:
  - `diagnostic_metrics_bootstrap.csv`

---

## D5. Diagnostic plots (Bootstrap)
- [ ] For each sector:
  - 2 transects × 3 points = 6 plots
- [ ] Each plot includes:
  - SIC(t)
  - ΔSIC(t)
  - SIC rank(t)
- [ ] Annotate plots with:
  - distance-to-edge
  - variability metrics
- [ ] Save to:
  - `results/diagnostics/transects/bootstrap/`

---

## D6. Behavioral classification (pre-phase)
- [ ] Define rule-based classes:
  - Monotonic / thermodynamic
  - Noisy-gradual
  - Impulse-dominated (MIZ-like)
- [ ] Assign class to each point
- [ ] Add column:
  - `behavior_class`
- [ ] Save:
  - `diagnostic_points_classified.csv`

---

## D7. AMSR-E replication (diagnostic only)
- [ ] Repeat **D1–D6** for AMSR-E where coverage allows
- [ ] Use AMSR-E *only* for:
  - resolving sub-grid variability
  - assessing impulse frequency/amplitude
- [ ] Explicitly **do not** compute long-term trends
- [ ] Save parallel outputs:
  - `diagnostic_metrics_amsre.csv`
  - `results/diagnostics/transects/amsre/`

---

## D8. Cross-sensor comparison (Bootstrap vs AMSR-E)
- [ ] Compare variability metrics at co-located points
- [ ] Assess:
  - impulse detectability
  - rank volatility differences
- [ ] Produce:
  - 1 comparison figure
  - 1 summary table

---

## D9. Optional: Vichi overlay (interpretive)
- [ ] Compute Vichi variability indicator
- [ ] Compare with:
  - distance bins
  - behavior classes
- [ ] Deliver:
  - 1 circumpolar map
  - 1 contingency table

---

## D10. Freeze diagnostics (stop here)
- [ ] Do **not**:
  - compute phase
  - map trends
  - optimize thresholds
- [ ] Diagnostics complete → rewrite questions
