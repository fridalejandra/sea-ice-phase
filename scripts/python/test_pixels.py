#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: SIC time series and daily change (ΔSIC) for a few pixels
in the Amundsen–Bellingshausen Sea (ABS).

- Uses the same input file and preprocessing as the dynamic-threshold scripts.
- For a chosen year, plots SIC and ΔSIC for 3 pixels:
    * ABS_coast      – near the coast
    * ABS_mid        – mid-shelf / marginal ice zone
    * ABS_offshore   – offshore open-ocean side

You MUST set the (y, x) indices for these three pixels yourself.
"""

import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
SENSOR     = "SMMR"
INPUT_FILE = "/user/geog/falejandraperez/sea-ice-phase/data/merged/merged_bootstrap_SH_latest.nc"
CONC_VAR   = "N07_ICECON"

OUT_DIR    = "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/ABS_transects"

# Year to inspect (start simple; you can make this a list later)
YEAR = 2005

# Pixels in ABS: fill these in with actual (y, x) indices
# Example placeholders; CHANGE THESE
PIXELS = {
    # closer to the continent (larger y)
    "ABS_coast":    {"y": 260, "x": 180},
    # marginal ice zone / mid-shelf
    "ABS_mid":      {"y": 230, "x": 180},
    # more open-ocean side (smaller y)
    "ABS_offshore": {"y": 200, "x": 180},
}


FEB29_MODE = "drop"  # standardize calendar the same way as detection code


# ---------------- HELPERS ---------------- #
def standardize_calendar(da: xr.DataArray, mode="drop"):
    """Drop Feb 29 to get a 365-day calendar per year."""
    if mode == "drop":
        return da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))
    if mode == "keep":
        return da
    raise ValueError(f"Unknown FEB29_MODE: {mode}")


def select_year(da: xr.DataArray, year: int) -> xr.DataArray:
    """Return a DataArray subset for the calendar year."""
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    return da.sel(time=slice(start, end))


def compute_dC(arr: np.ndarray):
    """Daily SIC change ΔSIC, aligned with time[1:]."""
    return np.diff(arr)


# ---------------- MAIN ---------------- #
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Opening SIC dataset: {INPUT_FILE}")
    ds = xr.open_dataset(INPUT_FILE)[[CONC_VAR, "x", "y", "time"]]

    ice = ds[CONC_VAR].astype("float32")
    # Scale to 0–1 if needed (same logic as dynamic script)
    if float(ice.max()) > 1.5:
        ice = ice / 100.0
    # Mask invalid values
    ice = ice.where(ice < 1.1)

    # Standardize calendar
    ice365 = standardize_calendar(ice, FEB29_MODE)

    # Extract year
    ice_year = select_year(ice365, YEAR)
    if ice_year.time.size == 0:
        raise RuntimeError(f"No data for YEAR={YEAR} after calendar standardization.")

    print(f"Year {YEAR}: {ice_year.time.size} days after standardization.")

    # Loop over pixels
    for name, idx in PIXELS.items():
        j = idx["y"]
        i = idx["x"]
        print(f"\n--- Pixel {name}: (y={j}, x={i}) ---")

        # Extract 1D time series
        ts = ice_year.isel(y=j, x=i)

        # If everything is NaN, skip
        if np.all(np.isnan(ts.values)):
            print("  All SIC values are NaN at this pixel/year – skipping.")
            continue

        # Drop NaNs for the diagnostic plot, but keep the original time index
        # We do this in a minimal way to avoid messing with alignment
        vals = ts.values.astype(float)
        t    = ts.time.values

        # Simple mask to avoid diff on NaN blocks
        mask = np.isfinite(vals)
        vals_valid = vals[mask]
        t_valid    = t[mask]

        if vals_valid.size < 3:
            print("  Too few valid SIC values to compute ΔSIC – skipping.")
            continue

        dC = compute_dC(vals_valid)
        t_mid = t_valid[1:]  # ΔC aligns from day 1..N-1

        # ------------- PLOT ------------- #
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # Panel 1: SIC time series
        ax1.plot(t_valid, vals_valid, lw=1)
        ax1.set_ylabel("SIC (0–1)")
        ax1.set_title(f"{name}  (y={j}, x={i}),  YEAR={YEAR}")

        # Panel 2: daily change ΔSIC
        ax2.plot(t_mid, dC, lw=1)
        ax2.axhline(0.0, lw=0.8, ls="--")
        ax2.set_ylabel("ΔSIC (per day)")
        ax2.set_xlabel("Date")

        fig.autofmt_xdate()
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"SIC_dC_{name}_Y{YEAR}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"  Wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
