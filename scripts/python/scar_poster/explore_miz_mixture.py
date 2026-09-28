"""
explore_miz_mixture.py   [EXPLORATORY -- option 3, Phase 1 only]

Did the concentration mixture within each sector actually shift post-2016?

The premise of the shift-share idea is that a sector is a mixture of
consolidated pack (low sensitivity) and marginal ice zone (high sensitivity),
and that the mixture moved. That premise has not been checked. This checks it
-- purely descriptively.

DESCRIPTIVE ONLY. No regression, no per-bin sensitivity, so no endogeneity
problem (we are not conditioning a response on a variable the response
affects; we are just describing how much area sits in each bin).

Uses sic_bootstrap_on_ease_sh.nc (regridded onto the EASE drift grid) and
assigns sectors by longitude band from ease_divergence_with_latlon.nc.
"""

import numpy as np
import pandas as pd
import xarray as xr

SIC_PATH = "sic_bootstrap_on_ease_sh.nc"
LATLON_PATH = "ease_divergence_with_latlon.nc"   # supplies 2D lon/lat

SPLIT_YEAR = 2016
EXCLUDE_YEARS = [1978, 1987, 1991, 1995]

# fixed bins, identical pre and post -- period-relative bins would build in
# the answer
BINS = [0.15, 0.40, 0.70, 0.90, 1.01]
BIN_LABELS = ["0.15-0.40 (open MIZ)", "0.40-0.70", "0.70-0.90",
              "0.90-1.00 (consolidated)"]

SECTORS = {
    "Amundsen-Bellingshausen": (230.0, 300.0),
    "Weddell":                 (300.0, 20.0),
    "King Haakon VII":         (20.0, 90.0),
    "East Antarctica":         (90.0, 160.0),
    "Ross-Amundsen":           (160.0, 230.0),
}

OUT = "explore_miz_mixture.csv"


def sector_of(lon2d):
    lab = np.full(lon2d.shape, "", dtype=object)
    lon360 = lon2d % 360.0
    for name, (lo, hi) in SECTORS.items():
        lo, hi = lo % 360.0, hi % 360.0
        m = ((lon360 >= lo) & (lon360 < hi)) if lo <= hi else \
            ((lon360 >= lo) | (lon360 < hi))
        lab[m] = name
    return lab


def main():
    sic = xr.open_dataset(SIC_PATH)["sic"]
    lon = xr.open_dataset(LATLON_PATH, decode_times=False)["lon"].values
    sec = sector_of(lon)

    yrs = sic["time"].dt.year
    sic = sic.sel(time=~yrs.isin(EXCLUDE_YEARS))
    post = (sic["time"].dt.year >= SPLIT_YEAR).values

    A = sic.values          # (t, y, x), 0-1, NaN where masked
    print(f"loaded {A.shape[0]} days")

    rows = []
    for name in SECTORS:
        m = (sec == name)
        for period, sel in (("pre", ~post), ("post", post)):
            a = A[sel][:, m]                       # (days, cells in sector)
            n_days = a.shape[0]
            # mean number of cells per day in each bin
            counts = []
            for i in range(len(BINS) - 1):
                lo, hi = BINS[i], BINS[i + 1]
                counts.append(np.nansum((a >= lo) & (a < hi)) / n_days)
            total = sum(counts)
            for lab, c in zip(BIN_LABELS, counts):
                rows.append({
                    "sector": name, "period": period, "bin": lab,
                    "mean_cells_per_day": c,
                    "fraction_of_ice_area": c / total if total else np.nan,
                })

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["sector", "bin"], columns="period",
                          values="fraction_of_ice_area")
    piv["change"] = piv["post"] - piv["pre"]
    res.to_csv(OUT, index=False)

    pd.set_option("display.width", 140)
    print("\nFraction of sector ice area in each concentration bin:")
    print(piv.round(4).to_string())
    print(f"\n-> {OUT}")
    print("\nRead: did the open/MIZ bins gain area share while the "
          "consolidated bin lost it? If the shift is large, the mixture "
          "premise holds and the full shift-share decomposition is worth "
          "doing. If the fractions barely move, the premise is wrong and "
          "the null needs a different explanation.")


if __name__ == "__main__":
    main()
