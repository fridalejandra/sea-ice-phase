"""
regrid_sic_to_ease.py

Regrid Bootstrap SIC (NSIDC polar stereographic, 332 x 316) onto the
EASE-Grid 2.0 South grid (321 x 321) used by the NSIDC-0116 drift/divergence
fields, so the two can be combined in the area budget.

WHY CONSERVATIVE, NOT BILINEAR
Sea ice concentration is area-intensive and the budget takes its spatial
derivative inside a flux. Conservative regridding preserves the area integral
(total ice area unchanged by the regrid); bilinear smears the ice edge, which
is exactly where the residual signal lives. Conservative is not optional here.

WHY REGRID SIC -> DRIFT (and not the reverse)
The entire divergence analysis already lives on the EASE grid. Moving SIC onto
that grid keeps every existing product consistent and touches nothing already
computed.

TIME-BOX: this is staged so an environment or coordinate failure shows up in
the first few minutes (Checkpoints 0-2), not after the weights compute. If you
hit a wall past Checkpoint 2, shelve it -- the residual can wait for Ch4.
"""

import sys

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
SIC_PATH = ("/user/geog/falejandraperez/sea-ice-phase/data/merged/"
            "SMMR_merged_19781101_20251231_complete.nc")
# use the explicit-date complete file for the real run; _latest.nc for a quick test

EASE_REF_PATH = ("/user/geog/falejandraperez/sea-ice-phase/scripts/python/"
                 "scar_poster/ice_divergence_daily_sh.nc")
# provides the TARGET grid (x/y and, ideally, 2D lat/lon). We only need its
# grid, not its data.

SIC_VAR = "N07_ICECON"     # from the ncdump; VERIFY -- there may be multiple
                            # sensor-specific concentration vars (N07_, F08_,
                            # etc). You want the merged/consistent one used by
                            # compute_sia.py. Check with Checkpoint 1.

OUT_PATH = "sic_bootstrap_on_ease_sh.nc"
REGRID_METHOD = "conservative"
WEIGHTS_PATH = "regrid_weights_bootstrap_to_ease.nc"   # cached after first build
# -----------------------------------------


def checkpoint(n, msg):
    print(f"\n{'='*60}\n[CHECKPOINT {n}] {msg}\n{'='*60}")


def main():
    # -- Checkpoint 0: is xesmf even importable? (fails in seconds if not) --
    checkpoint(0, "Import xesmf")
    try:
        import xesmf as xe
    except ImportError:
        print("xesmf not installed. Install with:")
        print("    conda install -c conda-forge xesmf")
        print("If that fights you, this is a SHELVE-IT signal for today.")
        sys.exit(1)
    print("xesmf imported OK.")

    # -- Checkpoint 1: open both, confirm variable + coordinate availability --
    checkpoint(1, "Open files, inspect variables and coordinates")
    sic = xr.open_dataset(SIC_PATH)
    ease = xr.open_dataset(EASE_REF_PATH)

    print("SIC data_vars:", list(sic.data_vars))
    print("SIC coords:   ", list(sic.coords))
    if SIC_VAR not in sic:
        print(f"\n[STOP] SIC_VAR {SIC_VAR!r} not found. Pick the right "
              f"concentration variable from the list above and re-run.")
        sys.exit(1)

    print("\nEASE (target) coords:", list(ease.coords))

    # conservative regridding needs CELL BOUNDS (lat_b/lon_b) or at least
    # 2D lat/lon cell centres it can infer bounds from. Check now.
    def has_latlon(ds):
        names = set(ds.coords) | set(ds.data_vars)
        lat = next((n for n in ("lat", "latitude", "TLAT") if n in names), None)
        lon = next((n for n in ("lon", "longitude", "TLON") if n in names), None)
        return lat, lon

    sic_lat, sic_lon = has_latlon(sic)
    ease_lat, ease_lon = has_latlon(ease)
    print(f"SIC lat/lon:  {sic_lat}, {sic_lon}")
    print(f"EASE lat/lon: {ease_lat}, {ease_lon}")

    if not (sic_lat and sic_lon):
        print("\n[STOP] SIC has no lat/lon coords. You'd need to reconstruct "
              "them from the polar-stereo projection (pyproj) before "
              "conservative regridding. That's extra scope -- SHELVE-IT signal.")
        sys.exit(1)
    if not (ease_lat and ease_lon):
        print("\n[WARN] EASE reference has no lat/lon. May need to reconstruct "
              "target grid lat/lon from x/y + EASE2 proj. Check before "
              "proceeding -- this adds scope.")

    # -- Checkpoint 2: assemble grids for xesmf --
    checkpoint(2, "Assemble source/target grids")
    sic_grid = xr.Dataset({
        "lat": sic[sic_lat],
        "lon": sic[sic_lon],
    })
    ease_grid = xr.Dataset({
        "lat": ease[ease_lat],
        "lon": ease[ease_lon],
    })
    print("Grids assembled. If conservative complains about missing bounds, "
          "fall back to REGRID_METHOD='bilinear' ONLY as a diagnostic to test "
          "the pipeline -- do not trust a bilinear result for the actual "
          "budget.")

    # -- Checkpoint 3: build (or load cached) regridder. The slow step. --
    checkpoint(3, "Build regridder (weights)")
    import os
    reuse = os.path.exists(WEIGHTS_PATH)
    regridder = xe.Regridder(
        sic_grid, ease_grid, REGRID_METHOD,
        filename=WEIGHTS_PATH, reuse_weights=reuse,
    )
    print(f"Regridder ready (weights {'reused' if reuse else 'built'}).")

    # -- Checkpoint 4: apply, one timestep first as a smoke test --
    checkpoint(4, "Apply to one timestep, sanity check")
    A = sic[SIC_VAR]
    if float(A.max()) > 1.5:
        print("Converting percent -> fraction.")
        A = A / 100.0

    first = A.isel(time=0)
    first_rg = regridder(first)
    print(f"One-timestep regrid: {first.shape} -> {first_rg.shape}")

    # -- Checkpoint 5: AREA CONSERVATION -- the check that matters --
    checkpoint(5, "Area conservation check")
    # crude cell-count-weighted proxy (equal-area target grid makes this fair):
    src_area = float(first.sum())
    dst_area = float(first_rg.sum())
    rel_err = abs(dst_area - src_area) / src_area if src_area else np.nan
    print(f"Sum(SIC) source={src_area:.1f}, target={dst_area:.1f}, "
          f"relative diff={rel_err:.3%}")
    print("On an EQUAL-AREA target grid, conservative regridding should give a "
          "small relative diff (a few % at most, from edge cells). A large "
          "diff means the grids or bounds are wrong -- STOP and diagnose.")

    # -- Checkpoint 6: full apply + write --
    checkpoint(6, "Regrid full record and write")
    A_rg = regridder(A)
    A_rg.name = "sic"
    A_rg.attrs["note"] = (
        f"Bootstrap SIC ({SIC_VAR}) conservatively regridded from NSIDC "
        f"polar-stereo onto EASE-Grid South to match NSIDC-0116 drift. "
        f"Method={REGRID_METHOD}."
    )
    A_rg.to_netcdf(OUT_PATH)
    print(f"-> {OUT_PATH}")
    print("\nNEXT: point test_budget_feasibility.py SIC_PATH at this file. "
          "Q1 (common grid) should now pass; proceed to the one-winter-month "
          "sanity test.")


if __name__ == "__main__":
    main()