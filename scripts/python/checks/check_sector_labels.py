#!/usr/bin/env python
"""check_sector_labels.py -- verify that sector-labelled ICE columns (SIE/SIA/extent) in a CSV belong to
the sector they are named after.  Counts ice cells (SIC >= 0.15) per sector_id of canonical_sectors.nc on the
15th of each month of 2015, converts to shares of the total, and matches each column's shares to the closest
sector_id.  Handles wide files (one column per sector) and long files (date, sector, value).
Non-ice files (winds) are reported as 'check by code'.  Usage: python check_sector_labels.py FILE.csv ..."""
import sys, re, numpy as np, pandas as pd, xarray as xr
ROOT = "/user/geog/falejandraperez/sea-ice-phase"
MASK = f"{ROOT}/data/canonical_sectors.nc"; SIC = f"{ROOT}/data/merged/merged_bootstrap_SH_latest.nc"
DATES = pd.date_range("2015-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)
KEYS = [("Amundsen_Bellingshausen", r"amund|belling|\babs\b"), ("Weddell", r"wedd|\bws\b"),
        ("King_Haakon", r"king|haakon|\bkh\b"), ("East_Antarctica", r"east|\bea\b"), ("Ross", r"ross|\bra\b")]
m = xr.open_dataset(MASK); sid = m["sector_id"]
names = {k: str(m[f"sector_{k}_name"].values) for k in range(1, 6)}
ds = xr.open_dataset(SIC); v = [x for x in ds.data_vars if x.endswith("_ICECON")][0]
grid = {}
for d in DATES:
    s = ds[v].sel(time=d, method="nearest"); ice = (s.where((s >= 0) & (s <= 1)) >= 0.15)
    c = {k: float(ice.where(sid == k).sum()) for k in range(1, 6)}; tot = sum(c.values())
    grid[d] = {k: c[k] / tot for k in c}
G = pd.DataFrame(grid).T
def label_of(col):
    for nm, pat in KEYS:
        if re.search(pat, col.lower().replace("-", "_").replace(" ", "_")): return nm
    return None
ICE = re.compile(r"sie|sia|extent|ice_area|icearea", re.I)
for f in sys.argv[1:]:
    print(f"\n=== {f}")
    d = pd.read_csv(f); tcol = [c for c in d.columns if c.lower() in ("time", "date")]
    if not tcol: print("  no time/date column -- annual or summary file, check by code"); continue
    d[tcol[0]] = pd.to_datetime(d[tcol[0]])
    scol = [c for c in d.columns if c.lower() in ("sector", "region", "code")]
    if scol:
        vcols = [c for c in d.columns if ICE.search(c) and pd.api.types.is_numeric_dtype(d[c])]
        if not vcols: print("  long format without an ice column -- check by code"); continue
        d = d[~d[scol[0]].astype(str).str.lower().str.contains("circum|total")]
        d = d.pivot_table(index=tcol[0], columns=scol[0], values=vcols[0])
        d.columns = [str(c) for c in d.columns]; print(f"  long format, ice column {vcols[0]}")
    else:
        if not ICE.search(f) and not any(ICE.search(c) for c in d.columns):
            print("  not an ice quantity -- share test does not apply, check by code"); continue
        d = d.set_index(tcol[0])
    cols = [c for c in d.columns if label_of(c) and "circum" not in c.lower() and "total" not in c.lower()]
    if len(cols) < 5: print(f"  sector columns found: {cols} -- need 5, check by hand"); continue
    sub = d[cols].reindex(DATES, method="nearest"); sub = sub.div(sub.sum(axis=1), axis=0); ok = True
    for c in cols:
        dist = {k: float(np.mean(np.abs(sub[c].values - G[k].values))) for k in range(1, 6)}
        k = min(dist, key=dist.get); true = names[k]
        match = label_of(c).split("_")[0].lower() in true.lower().replace("\u2013", "_").replace(" ", "_"); ok &= match
        print(f"  {c:32s} -> sector_id {k} ({true}); share diff {dist[k]:.3f}  {'OK' if match else '*** MISLABELLED ***'}")
    print("  ALL LABELS CORRECT" if ok else "  !!! LABELS DO NOT MATCH THE MASK !!!")
