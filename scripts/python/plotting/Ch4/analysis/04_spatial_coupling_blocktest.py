#!/usr/bin/env python
"""Spatial wind-divergence coupling at 75/100/150 km block sizes."""
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.formula.api as smf

REPO = "/user/geog/falejandraperez/sea-ice-phase"
DIV_PATH  = f"{REPO}/results/ch4/derived_nc/ease_divergence_with_latlon.nc"
WIND_PATH = f"{REPO}/results/ch4/derived_nc/wind_stress_on_ease_sh.nc"
DIV_VARS, WIND_VAR = ["div_positive", "div_negative"], "tau_mag"
X, Y, T = "x", "y", "time"

BLOCK_SIZES_KM = [100]
EASE_CELL_KM   = 25.0
REGIME_SHIFT_YEAR = 2016
SEASONS = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}
MIN_VALID_FRAC = 0.5
MIN_DAYS = 200
OUT_PREFIX = f"{REPO}/results/ch4/derived_nc/spatial_coupling"

def load(div_var):
    div = xr.open_dataset(DIV_PATH)[div_var]
    wind = xr.open_dataset(WIND_PATH)[WIND_VAR]
    # drop cruft coords BEFORE renaming, else the rename leaves dangling coords
    # on the old dim and coarsen collapses the array
    for extra in ["lat", "lon", "number", "expver"]:
        if extra in div.coords:
            div = div.drop_vars(extra)
        if extra in wind.coords:
            wind = wind.drop_vars(extra)
    if "valid_time" in wind.dims:
        wind = wind.rename({"valid_time": "time"})
    # div timestamps are midnight, wind timestamps are noon -- normalize both
    # to date-only before intersecting, else exact-datetime match finds nothing
    div = div.assign_coords(time=div["time"].dt.floor("D"))
    wind = wind.assign_coords(time=wind["time"].dt.floor("D"))
    t = np.intersect1d(div[T].values, wind[T].values)
    print(f"  common dates after normalizing time-of-day: {len(t)}", flush=True)
    return div.sel({T: t}), wind.sel({T: t})

def block_average(field, factor):
    mean = field.coarsen({X: factor, Y: factor}, boundary="trim").mean(skipna=True)
    frac = field.notnull().astype("float32").coarsen({X: factor, Y: factor}, boundary="trim").mean()
    return mean.where(frac >= MIN_VALID_FRAC)

def deseason(da):
    df = da.to_dataframe(name="v").reset_index()
    tt = pd.to_datetime(df[T])
    df["doy"] = tt.dt.dayofyear
    df["year"] = tt.dt.year
    df["period"] = np.where(df["year"] >= REGIME_SHIFT_YEAR, "post", "pre")
    df["anom"] = df["v"] - df.groupby(["period","doy"])["v"].transform("mean")
    return df["anom"].values, df["year"].values, tt.dt.month.values

def fit(div_anom, wind_anom, years, months, seas_months):
    m = np.isin(months, seas_months)
    d = pd.DataFrame({"d": div_anom[m], "w": wind_anom[m], "yr": years[m]}).dropna()
    if len(d) < MIN_DAYS:
        return None
    d["post"] = (d["yr"] >= REGIME_SHIFT_YEAR).astype(float)
    try:
        f = smf.ols("d ~ w + post + w:post", data=d).fit()
    except Exception:
        return None
    return (f.params.get("w", np.nan), f.params.get("w:post", np.nan),
            f.pvalues.get("w:post", np.nan))

def run():
  for div_var in DIV_VARS:
    print(f"\n#### variable: {div_var} ####", flush=True)
    div, wind = load(div_var)
    seas = list(SEASONS)
    for km in BLOCK_SIZES_KM:
        fac = int(round(km / EASE_CELL_KM))
        print(f"\n=== block {km} km (factor {fac}) ===", flush=True)
        print("DIV dims/coords:", div.dims, list(div.coords), div.shape, flush=True)
        print("WIND dims/coords:", wind.dims, list(wind.coords), wind.shape, flush=True)
        db = block_average(div, fac)
        print("DIV block_average OK", flush=True)
        wb = block_average(wind, fac)
        print("WIND block_average OK", flush=True)
        nyb, nxb = db.sizes[Y], db.sizes[X]
        b1 = np.full((len(seas), nyb, nxb), np.nan)
        b3 = np.full_like(b1, np.nan); p3 = np.full_like(b1, np.nan)
        for j in range(nyb):
            for i in range(nxb):
                dv, wv = db.isel({Y:j, X:i}), wb.isel({Y:j, X:i})
                if bool(dv.isnull().all()) or bool(wv.isnull().all()):
                    continue
                da_, yr, mo = deseason(dv)
                wa_, _, _   = deseason(wv)
                for s, sname in enumerate(seas):
                    r = fit(da_, wa_, yr, mo, SEASONS[sname])
                    if r is None: continue
                    b1[s,j,i], b3[s,j,i], p3[s,j,i] = r
            if j % 20 == 0:
                print(f"  row {j}/{nyb}", flush=True)
        out = xr.Dataset(
            {"beta1": (("season",Y,X), b1),
             "beta3": (("season",Y,X), b3),
             "p_beta3": (("season",Y,X), p3)},
            coords={"season": seas, "y": db[Y], "x": db[X]})
        path = f"{OUT_PREFIX}_block{km}km_{div_var}.nc"
        out.to_netcdf(path)
        print(f"  -> {path}  (fitted {int(np.isfinite(b3).sum())} block-seasons)", flush=True)

if __name__ == "__main__":
    run()
