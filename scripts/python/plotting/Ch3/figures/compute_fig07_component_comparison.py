#!/usr/bin/env python3
"""
compute_component_comparison.py -- Sect. 3.5: what the decomposition adds
beyond the SIE anomaly.

For each of the seven sector x index pairs (ch3_config.PRIMARY_PAIRS) the
same index, at the pair's season, is correlated with four summaries of that
sector's ice, all over 1979-2023 and all detrended:

  1. sie_anom   seasonal mean of the SIE anomaly from the invariant cycle
                (daily anomaly_from_iac averaged over the season) -- the
                traditional anomaly, i.e. the variable of Raphael and Hobbs
                (2014). Computed here from daily_fitted.csv for every season,
                so the East Antarctica / SAM_RET cell is no longer n/a.
  2. amplitude  observed amplitude anomaly; read from t35_index_scan_raw.csv
                (target amplitude_raw_anom) so it matches Fig. 8 exactly.
  3. max_doy    observed day-of-maximum anomaly; same source
                (target max_doy_raw_anom).
  4. raw_anom   seasonal mean of the raw APAC anomaly (daily residual_apac
                averaged over the season). Signed, like the others. This is
                the Sect. 3.4 quantity; the prediction from 3.4 is that this
                column is empty.
  (5. resid_sd  within-season SD of residual_apac -- kept in the CSV for
                reference, not plotted.)

Second block: the same four targets across the FULL scan (every sector in
t35_index_scan_raw.csv x every index-season column), reporting how many of
the cells reach p < 0.05 for each target, and the mean |r|. That is the
one-sentence version of the comparison for the text.

Seasons: DJF (Dec of Y-1 with Jan-Feb of Y, labelled Y), MAM, JJA, SON,
ADV = Mar-Aug, RET = Oct-Jan (Oct-Dec of Y with Jan of Y+1, labelled Y),
annual. These match compute_atmospheric_correlations.py.

Third block (if the ERA5 sector-wind CSV is present): the sector's own
seasonal-mean 10 m meridional wind (v, positive northward, i.e. a
southerly wind that pushes ice north and advects cold air) and wind speed
as local "indices" against the same four targets, in the advance and
retreat seasons and the annual mean. Rows go to t37c for panel (b) of the
heatmap.

Fourth block (same CSV): the daily-scale test. Day-to-day change in the
raw anomaly against the daily v-wind anomaly (day-of-year mean removed) at
lags 0-3 d, by sector, 1988-2023, with the same for the SIE anomaly.
Printed only.

Wind CSV: WIND_CSV env var, else <RAW_DATA_DIR>/ERA5_winds_daily_sector.csv
(from compute_ERA5_winds_daily_sector.py: columns time, u_<sector>,
v_<sector>, wind_<sector>).

Writes results/ch3/tables/t37_component_comparison.csv
       results/ch3/tables/t37b_component_comparison_scan.csv
       results/ch3/tables/t37c_wind_comparison.csv
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import (
    DAILY_CSV, INDEX_CSV, TABLES_DIR, PRIMARY_PAIRS, SECTORS_COMPUTE,
)
try:
    from ch3_config import RAW_DATA_DIR
except ImportError:
    RAW_DATA_DIR = os.path.join(HERE, "..", "data", "raw")
WIND_CSV = os.environ.get("WIND_CSV", os.path.join(RAW_DATA_DIR, "ERA5_winds_daily_sector.csv"))
WIND_SECTOR = {   # scan / daily sector label -> wind column suffix
    "Weddell": "Weddell", "ABS": "Amundsen_Bellingshausen",
    "Amundsen-Bellingshausen": "Amundsen_Bellingshausen", "Ross": "Ross",
    "East Antarctica": "East_Antarctica", "King Haakon": "King_Haakon",
    "Circumpolar": "circumpolar", "circumpolar": "circumpolar",
}

SCAN_PATH = os.path.join(TABLES_DIR, "t35_index_scan_raw.csv")
YEAR_LO, YEAR_HI = 1979, 2023   # index period

SEASON_MONTHS = {
    "DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11],
    "ADV": [3, 4, 5, 6, 7, 8], "RET": [10, 11, 12, 1], "annual": list(range(1, 13)),
}


def season_year(month, year, season):
    if season == "RET" and month == 1:
        return year - 1
    if season == "DJF" and month == 12:
        return year + 1
    return year


def detrend(x):
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return x
    t = np.arange(len(x), dtype=float)
    m, b = np.polyfit(t[ok], x[ok], 1)
    out = x.copy()
    out[ok] = x[ok] - (m * t[ok] + b)
    return out


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 6:
        return np.nan, np.nan, int(ok.sum())
    r, p = stats.pearsonr(x[ok], y[ok])
    return r, p, int(ok.sum())


# ── load ─────────────────────────────────────────────────────────────────────
scan = pd.read_csv(SCAN_PATH)
daily = pd.read_csv(DAILY_CSV, parse_dates=["Date"])
if "period" in daily.columns:
    daily = daily[daily["period"] == "FULL"]
daily = daily[(daily["Year"] >= YEAR_LO) & (daily["Year"] <= YEAR_HI)].copy()
daily["month"] = daily["Date"].dt.month
idx = pd.read_csv(INDEX_CSV)
idx = idx[(idx["Year"] >= YEAR_LO) & (idx["Year"] <= YEAR_HI)]

index_cols = [c for c in idx.columns
              if re.match(r"^(.*)_(annual|DJF|MAM|JJA|SON|ADV|RET)$", c)]

# seasonal means of the two daily quantities, per sector x season, cached
_season_cache = {}
_daily_labels = {str(x).lower(): x for x in daily["sector"].unique()}


def daily_for(sector_label):
    """daily rows for a scan sector label, tolerant of naming (ABS /
    SIE_Amundsen_Bellingshausen / circumpolar / Circumpolar ...)."""
    cands = [sector_label]
    inv = {v: k for k, v in SECTORS_COMPUTE.items()}
    cands += [inv.get(sector_label, ""), sector_label.replace(" ", "_"),
              "SIE_" + sector_label.replace(" ", "_")]
    if "circ" in sector_label.lower():
        cands += ["circumpolar", "Circumpolar", "SIE_circumpolar", "SIE_Circumpolar", "total"]
    for c in cands:
        if c and c.lower() in _daily_labels:
            return daily[daily["sector"] == _daily_labels[c.lower()]]
    return daily.iloc[0:0]


def seasonal_means(sector_label, season):
    key = (sector_label, season)
    if key in _season_cache:
        return _season_cache[key]
    d = daily_for(sector_label)
    d = d[d["month"].isin(SEASON_MONTHS[season])].copy()
    d["sy"] = [season_year(m, y, season) for m, y in zip(d["month"], d["Year"])]
    g = d.groupby("sy")
    out = pd.DataFrame({
        "Year": g.size().index,
        "sie_anom": g["anomaly_from_iac"].mean().values,
        "raw_anom": g["residual_apac"].mean().values,
        "resid_sd": g["residual_apac"].std().values,
        "ndays": g.size().values,
    })
    # drop partial seasons at the record end (e.g. RET 2023 without Jan 2024).
    # 1979-1987 (SMMR) is sampled every other day, so a season then has about
    # half the days of a season after 1987; the threshold must let those
    # through or the SIE-anomaly and raw-anomaly columns silently become
    # 1988-2023 while amplitude and day of maximum stay 1979-2023.
    full = out["ndays"] >= 0.35 * out["ndays"].median()
    out = out[full & (out["Year"] >= YEAR_LO) & (out["Year"] <= YEAR_HI)]
    _season_cache[key] = out
    return out


from ch3_config import ANNUAL_CSV
_ann = pd.read_csv(ANNUAL_CSV)
if "period" in _ann.columns:
    _ann = _ann[_ann["period"] == "FULL"]


def annual_scalars(sector_label):
    a = _ann[_ann["sector"] == sector_label]
    if a.empty:
        inv = {v: k for k, v in SECTORS_COMPUTE.items()}
        a = _ann[_ann["sector"] == inv.get(sector_label, sector_label)]
    a = a[(a["Year"] >= YEAR_LO) & (a["Year"] <= YEAR_HI)]
    return a[["Year", "amplitude_raw_anom", "max_doy_raw_anom"]].copy()


def scan_lookup(sector_label, index_col, target):
    h = scan[(scan["sector"] == sector_label) & (scan["index"] == index_col) & (scan["target"] == target)]
    if h.empty:
        return np.nan, np.nan, 0
    return float(h.iloc[0]["r"]), float(h.iloc[0]["p"]), int(h.iloc[0]["n"])


def one_cell(sector_label, index_col, predictor=None):
    """predictor: optional DataFrame(Year, <index_col>) replacing the index
    table (used for the local wind rows); amplitude / max_doy then come from
    annual_params via `annual` rather than the scan."""
    base, season = re.match(r"^(.*)_(annual|DJF|MAM|JJA|SON|ADV|RET)$", index_col).groups()
    src = idx[["Year", index_col]] if predictor is None else predictor
    sm = seasonal_means(sector_label, season).merge(src, on="Year")
    if predictor is None:
        r_amp, p_amp, _ = scan_lookup(sector_label, index_col, "amplitude_raw_anom")
        r_max, p_max, _ = scan_lookup(sector_label, index_col, "max_doy_raw_anom")
    else:
        a = annual_scalars(sector_label).merge(src, on="Year")
        r_amp, p_amp, _ = corr(a[index_col], a["amplitude_raw_anom"])
        r_max, p_max, _ = corr(a[index_col], a["max_doy_raw_anom"])
    r_sie, p_sie, n = corr(sm[index_col], detrend(sm["sie_anom"]))
    r_raw, p_raw, _ = corr(sm[index_col], detrend(sm["raw_anom"]))
    r_sd, p_sd, _ = corr(sm[index_col], detrend(sm["resid_sd"]))
    return dict(sector=sector_label, index=index_col, index_base=base, season=season, n=n,
                r_sie_anom=r_sie, p_sie_anom=p_sie,
                r_amplitude=r_amp, p_amplitude=p_amp,
                r_max_doy=r_max, p_max_doy=p_max,
                r_raw_anom=r_raw, p_raw_anom=p_raw,
                r_resid_sd=r_sd, p_resid_sd=p_sd)


# ── block 1: the seven pairs ─────────────────────────────────────────────────
rows = []
for sie_sector, target_var, index_col, basis in PRIMARY_PAIRS:
    lab = SECTORS_COMPUTE.get(sie_sector, sie_sector)
    c = one_cell(lab, index_col)
    c["basis"] = basis
    c["pair_target"] = target_var
    rows.append(c)
    f = lambda r: "  n/a " if pd.isna(r) else f"{r:+.2f}"
    print(f"{lab:16s} {index_col:14s} n={c['n']:2d}  SIE anom {f(c['r_sie_anom'])}  "
          f"amplitude {f(c['r_amplitude'])}  day of max {f(c['r_max_doy'])}  "
          f"raw anomaly {f(c['r_raw_anom'])}  (resid SD {f(c['r_resid_sd'])})")
out = pd.DataFrame(rows)
p1 = os.path.join(TABLES_DIR, "t37_component_comparison.csv")
out.to_csv(p1, index=False)
print(f"wrote {p1}")

# ── block 2: the full scan ───────────────────────────────────────────────────
sectors = sorted(scan["sector"].unique())
cells = []
for sec in sectors:
    for ic in index_cols:
        try:
            cells.append(one_cell(sec, ic))
        except Exception as e:  # a sector missing from daily, etc.
            print(f"  skip {sec} {ic}: {e}")
sc = pd.DataFrame(cells)
p2 = os.path.join(TABLES_DIR, "t37b_component_comparison_scan.csv")
sc.to_csv(p2, index=False)
print(f"wrote {p2}  ({len(sc)} cells x 4 targets)")

print("\nFull scan, all sectors x index-seasons: cells with p < 0.05 and mean |r|")
for tgt, name in [("sie_anom", "seasonal SIE anomaly"), ("amplitude", "amplitude"),
                  ("max_doy", "day of maximum"), ("raw_anom", "raw anomaly (seasonal mean)"),
                  ("resid_sd", "raw anomaly SD (reference)")]:
    r = sc[f"r_{tgt}"]; p = sc[f"p_{tgt}"]
    n = p.notna().sum()
    print(f"  {name:30s} {int((p < 0.05).sum()):3d} of {n:3d} at p<0.05 "
          f"(expected {0.05 * n:.0f});  mean |r| = {np.nanmean(np.abs(r)):.3f};  "
          f"max |r| = {np.nanmax(np.abs(r)):.2f}")

print("\nPer sector, strongest |r| for each target:")
for sec in sectors:
    s = sc[sc["sector"] == sec]
    line = f"  {sec:16s}"
    for tgt in ["sie_anom", "amplitude", "max_doy", "raw_anom"]:
        if s[f"r_{tgt}"].notna().sum() == 0:
            line += f"  {tgt}: n/a"
            continue
        i = s[f"r_{tgt}"].abs().idxmax()
        line += f"  {tgt}: {s.loc[i, f'r_{tgt}']:+.2f} ({s.loc[i, 'index']})"
    print(line)


# ── block 3 and 4: local wind ────────────────────────────────────────────────
if not os.path.exists(WIND_CSV):
    print(f"\n(no wind CSV at {WIND_CSV}; skipping wind blocks -- set WIND_CSV to add them)")
    sys.exit(0)

w = pd.read_csv(WIND_CSV)
raw_time = w["time"].astype(str)
# the file may hold the repr of a DatetimeIndex ("DatetimeIndex(['1979-01-01 12:00:00'], ...)");
# pull the first YYYY-MM-DD out of whatever the string is
t = pd.to_datetime(raw_time.str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce", format="%Y-%m-%d")
if t.isna().mean() > 0.5:
    t = pd.to_datetime(raw_time, errors="coerce")
if t.isna().mean() > 0.5:
    sys.exit(f"could not parse the wind time column; first values: {raw_time.head(3).tolist()}")
w["time"] = t
w = w.dropna(subset=["time"])
# accept u10_/v10_ (ERA5 variable names) as well as u_/v_
w = w.rename(columns={c: c.replace("u10_", "u_").replace("v10_", "v_") for c in w.columns})
if not any(c.startswith("v_") for c in w.columns):
    print(f"\n{WIND_CSV} has no v_<sector> columns (columns: {list(w.columns)[:6]}...);\n"
          "it is the wind-SPEED file. Run compute_ERA5_winds_daily_sector.py on the cluster "
          "for the signed u, v version, then rerun. Skipping wind blocks.")
    sys.exit(0)
w["Year"] = w["time"].dt.year
w["month"] = w["time"].dt.month
w["doy"] = w["time"].dt.dayofyear
print(f"\nwind: {WIND_CSV}  {w['time'].min().date()} to {w['time'].max().date()}, {len(w)} days")
print("wind sanity (units are whatever the producing script wrote; a sector MEAN should be a few m/s,")
print("a SUM over the mask thousands -- either is fine for correlation; 'wind' should be >0 if it is a speed):")
for suf in sorted({c[2:] for c in w.columns if c.startswith("v_")}):
    u_, v_ = w.get(f"u_{suf}"), w.get(f"v_{suf}")
    sp = w.get(f"wind_{suf}")
    line = f"  {suf:26s} u mean {u_.mean():9.1f} sd {u_.std():8.1f}   v mean {v_.mean():9.1f} sd {v_.std():8.1f}"
    if sp is not None and u_ is not None:
        est = np.sqrt(u_ ** 2 + v_ ** 2)
        ok = ~(est.isna() | sp.isna())
        rr = np.corrcoef(est[ok], sp[ok])[0, 1] if ok.sum() > 10 else np.nan
        line += f"   wind min {sp.min():9.1f}  corr(wind, sqrt(u2+v2)) {rr:+.2f}"
    print(line)


def wind_seasonal(sector_label, var, season):
    col = f"{var}_{WIND_SECTOR[sector_label]}"
    d = w[w["month"].isin(SEASON_MONTHS[season])].copy()
    d["sy"] = [season_year(m, y, season) for m, y in zip(d["month"], d["Year"])]
    g = d.groupby("sy")[col]
    out = pd.DataFrame({"Year": g.mean().index, "val": g.mean().values, "n": g.size().values})
    out = out[(out["n"] >= 0.8 * out["n"].median()) & (out["Year"] >= YEAR_LO) & (out["Year"] <= YEAR_HI)]
    out["val"] = detrend(out["val"].values)
    return out[["Year", "val"]]


wind_rows = []
_pref = ["Weddell", "King Haakon", "East Antarctica", "Ross", "ABS", "Amundsen-Bellingshausen"]
wind_sectors = [s_ for s_ in _pref if s_ in sectors] + [s_ for s_ in sectors if s_ not in _pref]
wind_sectors = [s_ for s_ in wind_sectors if s_ in WIND_SECTOR and f"v_{WIND_SECTOR[s_]}" in w.columns]
print("\nLocal wind (seasonal mean, detrended) against the four targets:")
for sec in wind_sectors:
    for var, vname in (("v", "v-wind"), ("wind", "speed")):
        for season in ("ADV", "RET", "annual"):
            name = f"{var}wind_{season}"
            pred = wind_seasonal(sec, var, season).rename(columns={"val": name})
            c = one_cell(sec, name, predictor=pred)
            c["index_base"] = vname
            wind_rows.append(c)
            f = lambda r: "  n/a " if pd.isna(r) else f"{r:+.2f}"
            print(f"  {sec:16s} {vname:7s} {season:6s} n={c['n']:2d}  SIE anom {f(c['r_sie_anom'])}  "
                  f"amplitude {f(c['r_amplitude'])}  day of max {f(c['r_max_doy'])}  raw anomaly {f(c['r_raw_anom'])}")
wr = pd.DataFrame(wind_rows)
p3 = os.path.join(TABLES_DIR, "t37c_wind_comparison.csv")
wr.to_csv(p3, index=False)
print(f"wrote {p3}")

# daily scale: tendency of the raw anomaly vs daily v anomaly, lags 0-3
print("\nDaily scale, 1988-2023: r between the day-to-day change of the raw anomaly")
print("(and of the SIE anomaly) and the v-wind anomaly LAG days earlier, by sector")
for sec in wind_sectors:
    col = f"v_{WIND_SECTOR[sec]}"
    ws = w[["time", col, "doy"]].copy()
    ws["Date"] = ws["time"].dt.normalize()   # 12 UTC stamps -> calendar day
    ws["v_anom"] = ws[col] - ws.groupby("doy")[col].transform("mean")
    d = daily_for(sec)
    d = d[d["Year"] >= 1988].sort_values("Date")
    d = d.merge(ws[["Date", "v_anom"]], on="Date", how="inner")
    d["d_raw"] = d["residual_apac"].diff()
    d["d_sie"] = d["anomaly_from_iac"].diff()
    line = f"  {sec:16s}"
    for lag in range(4):
        vlag = d["v_anom"].shift(lag)
        r1, _, n = corr(vlag, d["d_raw"])
        r2, _, _ = corr(vlag, d["d_sie"])
        line += f"  lag{lag}: raw {r1:+.2f} sie {r2:+.2f}"
    print(line + f"   (n={n})")