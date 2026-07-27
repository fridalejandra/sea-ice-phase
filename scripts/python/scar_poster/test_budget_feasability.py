"""
test_budget_feasibility.py

FIVE-MINUTE GO/NO-GO before committing a block to the area budget.

Answers two questions in order:
  Q1. Are the drift and SIC products on a common grid? (If no, a regridding
      step is required, and that is its own block.)
  Q2. On ONE MONTH of data, does the budget behave sanely?

Run this before compute_area_budget_residual.py. Testing on one month rather
than 45 years means a failure costs minutes, not hours.
"""

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
DRIFT_PATH = "TODO_nsidc0116_drift.nc"
SIC_PATH = "TODO_sic_gridded.nc"

# one winter month, pack well established, low melt -- the cleanest possible
# test case. If the budget fails HERE it will fail everywhere.
TEST_START = "2010-07-01"
TEST_END = "2010-07-31"
# -----------------------------------------


def inspect(path, label):
    ds = xr.open_dataset(path)
    print(f"\n--- {label} ---")
    print("  data_vars:", list(ds.data_vars))
    print("  coords:   ", list(ds.coords))
    print("  sizes:    ", dict(ds.sizes))

    for c in ("x", "y", "xgrid", "ygrid", "latitude", "longitude", "lat", "lon"):
        if c in ds.coords:
            v = ds[c].values
            if v.ndim == 1 and v.size > 1:
                print(f"  {c}: {v[0]:.1f} .. {v[-1]:.1f}, "
                      f"spacing {np.diff(v)[0]:.3f}, n={v.size}")

    for a in ("grid_mapping", "grid_mapping_name", "proj4text", "spatial_ref",
              "crs", "projection"):
        if a in ds.attrs:
            print(f"  attr {a}: {ds.attrs[a]}")
        if a in ds.variables:
            print(f"  var {a}.attrs: {dict(ds[a].attrs)}")

    return ds


def main():
    drift = inspect(DRIFT_PATH, "DRIFT")
    sic = inspect(SIC_PATH, "SIC")

    print("\n=== Q1: COMMON GRID? ===")
    dspatial = {d: s for d, s in drift.sizes.items() if d != "time"}
    sspatial = {d: s for d, s in sic.sizes.items() if d != "time"}

    if dspatial == sspatial:
        print(f"  Dimension sizes MATCH: {dspatial}")
        print("  Now check the projection metadata printed above by eye.")
        print("  NSIDC-0116 is typically EASE-Grid 2.0; NSIDC SIC products")
        print("  (0051/0079/G02202) are typically polar stereographic.")
        print("  MATCHING SIZES DO NOT PROVE MATCHING PROJECTIONS.")
        print("  -> If projections agree: GO, no regridding needed.")
        print("  -> If they differ: STOP, regridding block required.")
    else:
        print(f"  Dimension sizes DIFFER: drift={dspatial}, sic={sspatial}")
        print("  -> REGRIDDING REQUIRED. Budget a full block for this before")
        print("     any budget computation. This is the schedule decision.")
        return

    print("\n=== Q2: ONE-MONTH SANITY TEST ===")
    print(f"  Slicing {TEST_START} to {TEST_END}")
    print("  Run compute_area_budget_residual.py on this slice only, then")
    print("  check, in this order:")
    print("    1. Winter pack interior (A>0.95) mean |residual| -- should be")
    print("       small. This is the single most diagnostic number. If it is")
    print("       comparable to the melt-season signal you hope to detect,")
    print("       the dynamic term is wrong and the residual is meaningless.")
    print("    2. Fraction of domain-days flagged for ridging. Small = ignore")
    print("       and caveat. Large = a real modelling decision, not a footnote.")
    print("    3. PLOT the spatial mean |residual|. Following the ice edge is")
    print("       plausible physics. Following grid seams or the land mask is")
    print("       a bug. This one is judged by eye, not by a number.")
    print("\n  If all three pass on one winter month: GO for the full record.")
    print("  If any fails: the residual does not belong on the poster, and")
    print("  that is a finding about data limits, not a failure.")


if __name__ == "__main__":
    main()