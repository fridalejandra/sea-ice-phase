"""
test_budget_onemonth.py

FAST feasibility test of the sea ice area budget on ONE winter month, using
the approximate dynamic term A*(del.u) -- the divergence you already have,
times concentration -- rather than the full flux del.(u*A). No raw drift
vectors needed.

    dA/dt  ~=  A * (del.u)   +   residual
               \___________/     \______/
                approx dynamic     thermodynamic + (the u.grad(A) term this
                                    approximation drops)

WHAT THIS TESTS
Whether the budget CLOSES well enough to be worth pursuing. The single most
diagnostic number is the residual in the deep winter pack interior (A ~ 1),
where little thermodynamic change happens and the dynamic term should nearly
account for dA/dt on its own. If the residual there is comparable to the
melt-season signal you'd hope to detect, the budget (in this cheap form) is
too noisy to use -- and you'd learn that in two minutes instead of after
loading and regridding the full drift record.

IMPORTANT CAVEAT
This drops the advection term u.grad(A), which is largest at the ice edge.
So expect the residual to be worse at the edge than in the interior even if
everything is working. The interior check is the fair one. A promising
interior result justifies building the full flux form; a bad interior result
means don't bother.
"""

import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
DIV_PATH = "ice_divergence_daily_sh.nc"          # has 'divergence' on EASE grid
SIC_PATH = "sic_bootstrap_on_ease_sh.nc"         # regridded SIC on same grid

DIV_VAR = "divergence"
SIC_VAR = "sic"

# divergence units: the compute script noted central differences with dx in
# metres and drift in cm/s or m/s -- VERIFY. If divergence is per-second,
# multiply by 86400 to get per-day. Check the printed magnitude below.
DIV_TO_PER_DAY = 86400.0     # set to 1.0 if divergence is already per-day

TEST_START = "2010-07-01"
TEST_END = "2010-07-31"

A_INTERIOR = 0.95            # deep pack threshold for the interior check
A_EDGE_LO, A_EDGE_HI = 0.15, 0.80   # "ice edge zone" band
# -----------------------------------------


def main():
    div = xr.open_dataset(DIV_PATH)
    sic = xr.open_dataset(SIC_PATH)

    print("div vars:", list(div.data_vars), "| dims:", dict(div.sizes))
    print("sic vars:", list(sic.data_vars), "| dims:", dict(sic.sizes))

    D = div[DIV_VAR]
    A = sic[SIC_VAR]


    # align time and slice the test month
    D = D.sel(time=slice(TEST_START, TEST_END))
    A = A.sel(time=slice(TEST_START, TEST_END))

    # ensure the two share timestamps
    common = np.intersect1d(D["time"].values, A["time"].values)
    if len(common) == 0:
        print("[STOP] No overlapping timestamps in the test month. Check the "
              "time coordinates/calendars of the two files.")
        return
    D = D.sel(time=common)
    A = A.sel(time=common)
    print(f"\nTest month: {len(common)} shared days "
          f"({str(common[0])[:10]} .. {str(common[-1])[:10]})")

    # report divergence magnitude so units can be sanity-checked
    dmag = float(np.nanmedian(np.abs(D.values)))
    print(f"median |divergence| = {dmag:.3e} (raw units). After x{DIV_TO_PER_DAY:g}"
          f" -> {dmag*DIV_TO_PER_DAY:.3e} /day. Sea ice divergence is typically "
          f"~1e-2 to 1e-1 /day; if this is wildly off, fix DIV_TO_PER_DAY.")

    # approximate dynamic term
    dyn = A * D * DIV_TO_PER_DAY          # /day
    # note sign: divergence positive = opening = ice area should DECREASE
    # locally, so dA/dt ~ -A*(del.u). We'll fit sign empirically below rather
    # than assume, but flag the convention.

    # dA/dt by centred difference in time (per day)
    dAdt = (A.shift(time=-1) - A.shift(time=1)) / 2.0

    # residual under BOTH sign conventions, so we don't guess wrong
    resid_minus = dAdt - (-dyn)   # dA/dt = -A*div + resid
    resid_plus = dAdt - (dyn)     # dA/dt = +A*div + resid

    def interior_stats(resid):
        r = resid.where(A > A_INTERIOR)
        return float(np.nanmean(np.abs(r.values)))

    im = interior_stats(resid_minus)
    ip = interior_stats(resid_plus)
    print(f"\nWinter pack-interior (A>{A_INTERIOR}) mean |residual|:")
    print(f"   dA/dt = -A*div + resid :  {im:.4e} /day")
    print(f"   dA/dt = +A*div + resid :  {ip:.4e} /day")
    better = "minus (-A*div)" if im < ip else "plus (+A*div)"
    resid = resid_minus if im < ip else resid_plus
    print(f"   -> better sign convention: {better}")

    # scale reference: how big is dA/dt itself in the interior?
    dadt_interior = float(np.nanmean(np.abs(dAdt.where(A > A_INTERIOR).values)))
    dadt_edge = float(np.nanmean(np.abs(
        dAdt.where((A > A_EDGE_LO) & (A < A_EDGE_HI)).values)))
    print(f"\nFor scale:")
    print(f"   mean |dA/dt| interior (A>{A_INTERIOR}):     {dadt_interior:.4e} /day")
    print(f"   mean |dA/dt| edge zone ({A_EDGE_LO}-{A_EDGE_HI}): {dadt_edge:.4e} /day")

    best_interior = min(im, ip)
    ratio = best_interior / dadt_edge if dadt_edge else np.nan
    print(f"\n=== VERDICT ===")
    print(f"interior |residual| / edge |dA/dt| = {ratio:.2f}")
    if ratio < 0.3:
        print("PROMISING: interior residual is small relative to the edge-zone "
              "signal you'd want to detect. The budget closes reasonably in "
              "the interior -> worth building the full flux form with raw u/v.")
    elif ratio < 0.7:
        print("MARGINAL: interior residual is a noticeable fraction of the "
              "signal. The full flux form (adding u.grad(A)) might tighten it, "
              "but no guarantee. Judgement call given your timeline.")
    else:
        print("POOR: interior residual is comparable to the signal. In this "
              "cheap form the budget is too noisy to use. The full flux form "
              "could still help, but this is a SHELVE-FOR-POSTER signal.")

    # edge-zone residual for context (expected worse -- advection dropped)
    edge_resid = float(np.nanmean(np.abs(
        resid.where((A > A_EDGE_LO) & (A < A_EDGE_HI)).values)))
    print(f"\n(Edge-zone mean |residual| = {edge_resid:.4e} /day -- expected to "
          f"be worse here since this approximation drops the ice-edge "
          f"advection term. Don't judge the budget on this number.)")


if __name__ == "__main__":
    main()