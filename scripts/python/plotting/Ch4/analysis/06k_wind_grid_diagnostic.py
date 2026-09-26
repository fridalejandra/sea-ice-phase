"""
06k_wind_grid_diagnostic.py -- is wind_stress_on_ease_sh.nc actually on the same
grid, in the same orientation, and on the same time axis as the ice divergence
file?  A square grid (321x321) means a transposed or flipped wind array passes a
shape check and correlates with nothing -- which is what 06i/06j returned.

Checks
  1. Structure: dims/coords/attrs of the wind file; time monotonic & unique;
     whether it carries its own lat/lon and, if so, how they compare to the ice
     file's lat/lon (as-is and transposed).
  2. Physics: mean tau_x in the 55-62S band (westerlies -> should be > 0) and in
     the 70-78S band (coastal easterlies -> should be < 0), for four orientations
     of the wind array: as-is, transposed, y-flipped, x-flipped.
  3. Cross-check against the paper's own sector wind-stress series: daily
     sector-mean |tau| from this file (Weddell, ice-file sector mask) vs the
     wind_stress column for Weddell in analysis_table_daily_anomaly_clean.csv.
     r should be ~0.9+ if the grid mapping is right.
"""
import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/user/geog/falejandraperez/sea-ice-phase"
ICE_NC = f"{ROOT}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
WIND_NC = f"{ROOT}/results/ch4/derived_nc/wind_stress_on_ease_sh.nc"
TABLE = f"{ROOT}/data/merged/analysis_table_daily_anomaly_clean.csv"
STEP = 7            # subsample every STEP-th day for the band means (speed)


def time_name(ds):
    for c in ("time", "valid_time"):
        if c in ds.dims:
            return c
    raise KeyError(list(ds.dims))


def main():
    ice = xr.open_dataset(ICE_NC)
    wind = xr.open_dataset(WIND_NC)
    wt = time_name(wind)
    print("=== 1. STRUCTURE ===")
    print(wind)
    print("\nwind dims order for tau_x:", wind["tau_x"].dims)
    print("ice  dims order for divergence:", ice["divergence"].dims)
    t = pd.DatetimeIndex(wind[wt].values)
    print(f"wind time: {t[0]} -> {t[-1]}, n={len(t)}, monotonic={t.is_monotonic_increasing}, "
          f"unique={t.is_unique}")
    for c in ("x", "y"):
        if c in wind.coords and c in ice.coords:
            same = np.allclose(wind[c].values, ice[c].values)
            print(f"coord {c}: identical to ice file = {same}; "
                  f"wind[{c}][:3]={wind[c].values[:3]}, ice[{c}][:3]={ice[c].values[:3]}")
    lat_i = ice["lat"].values; lon_i = ice["lon"].values
    for name in ("lat", "latitude", "lon", "longitude"):
        if name in wind.variables:
            w = wind[name].values
            if w.shape == lat_i.shape:
                ref = lat_i if name.startswith("lat") else lon_i
                print(f"wind {name}: max|diff| vs ice as-is = {np.nanmax(np.abs(w - ref)):.3g}, "
                      f"transposed = {np.nanmax(np.abs(w.T - ref)):.3g}")

    print("\n=== 2. PHYSICS: zonal-band mean tau_x under four orientations ===")
    idx = np.arange(0, len(t), STEP)
    tx = wind["tau_x"].isel({wt: idx}).values.astype("float64")
    tx_mean = np.nanmean(tx, axis=0)                      # (y, x) as stored
    west = (lat_i <= -55) & (lat_i >= -62)
    east = (lat_i <= -70) & (lat_i >= -78)
    orients = {"as-is": tx_mean, "transposed": tx_mean.T,
               "y-flipped": tx_mean[::-1, :], "x-flipped": tx_mean[:, ::-1]}
    print(f"{'orientation':12s} {'55-62S mean tau_x':>18s} {'70-78S mean tau_x':>18s}   verdict")
    for k, a in orients.items():
        w_ = np.nanmean(a[west]); e_ = np.nanmean(a[east])
        ok = (w_ > 0) and (e_ < 0)
        print(f"{k:12s} {w_:18.4g} {e_:18.4g}   {'PASS' if ok else 'fail'}")
    print("(expect PASS for exactly one orientation: westerlies positive, coastal easterlies negative)")

    print("\n=== 3. CROSS-CHECK vs paper's Weddell sector wind stress ===")
    tab = pd.read_csv(TABLE, parse_dates=["date"])
    ws = tab[tab.sector.str.contains("Wed", case=False)].set_index("date")["wind_stress"].sort_index()
    lon = ((lon_i + 180) % 360) - 180
    wmask = (lon >= -60) & (lon < 20) & (lat_i <= -55)
    common = pd.DatetimeIndex(t.normalize()).intersection(ws.index)
    common = common[common.year >= 1988]
    iw = np.searchsorted(pd.DatetimeIndex(t.normalize()), common)
    txw = wind["tau_x"].isel({wt: iw}).values; tyw = wind["tau_y"].isel({wt: iw}).values
    mag = np.hypot(txw, tyw)
    for k, sl in {"as-is": (slice(None), slice(None)), "transposed": "T",
                  "y-flipped": (slice(None, None, -1), slice(None)),
                  "x-flipped": (slice(None), slice(None, None, -1))}.items():
        m = np.transpose(mag, (0, 2, 1)) if sl == "T" else mag[(slice(None),) + sl]
        series = np.nanmean(m[:, wmask], axis=1)
        r = np.corrcoef(series, ws.loc[common].values)[0, 1]
        ratio = np.nanmedian(series) / np.nanmedian(ws.loc[common].values)
        print(f"{k:12s} r(daily |tau| this file vs table) = {r:.3f}   median ratio file/table = {ratio:.3g}")
    print("(expect r ~0.9+ for the correct orientation; ratio ~1/24 if this file still has the /86400 bug)")


if __name__ == "__main__":
    main()
