#!/usr/bin/env python3
"""
check_wind_sign.py -- is the sector-mean wind file what we think it is?

compute_wind_response.py finds the ice-edge response to the meridional wind
REVERSED in the Weddell and King Haakon sectors (deflection ~ -150 to -170 deg)
relative to the Pacific sectors. Before reading that as physics, rule out the
file. Three checks, none needing ERA5 itself:

 1. Provenance: the speed column must be >= 0 (it is sqrt(u^2+v^2) in the
    script on disk). Negative values mean the CSV was written by an older
    version of compute_ERA5_winds_daily_sector.py, and the u/v columns are then
    of unknown convention. Also prints file dates.
 2. Components are geographic, not grid: the 1988-2023 mean u must be clearly
    positive (westerlies) in every sector, with |mean v| small. Vectors rotated
    into polar-stereographic grid axes would give mean "u" near zero or
    negative and a large mean "v" in sectors far from the grid's y axis.
 3. Coherence: v in adjacent sectors should correlate positively (synoptic
    systems span ~20-30 deg of longitude). A sector whose v is anticorrelated
    with both neighbours has a sign problem of its own.

If all three pass, the v columns are geographic northward wind with the same
convention in every sector, and the Atlantic reversal is in the ice, not the file.

Usage: python check_wind_sign.py            (WIND_CSV env var overrides the path)
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from ch3_config import RAW_DATA_DIR
except Exception:
    RAW_DATA_DIR = os.path.join(HERE, "..", "data", "raw")
WIND = os.environ.get("WIND_CSV", os.path.join(RAW_DATA_DIR, "ERA5_winds_daily_sector.csv"))
SCRIPT = os.path.join(HERE, "compute_ERA5_winds_daily_sector.py")

# west-to-east order around the continent, so neighbours are adjacent
ORDER = ["Weddell", "King_Haakon", "East_Antarctica", "Ross", "Amundsen_Bellingshausen"]

w = pd.read_csv(WIND)
# the extraction wrote u10_/v10_; compute_wind_response.py renames them the same way
w = w.rename(columns={c: c.replace("u10_", "u_").replace("v10_", "v_") for c in w.columns})
tcol = next(c for c in w.columns if c.lower() in ("time", "date", "datetime"))
w[tcol] = pd.to_datetime(w[tcol].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
w = w[(w[tcol].dt.year >= 1988) & (w[tcol].dt.year <= 2023)]
print(f"wind file: {WIND}\n  rows {len(w)}, {w[tcol].min().date()} to {w[tcol].max().date()}")
if os.path.exists(SCRIPT):
    ts = lambda p: time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
    print(f"  CSV modified {ts(WIND)};  compute_ERA5_winds_daily_sector.py modified {ts(SCRIPT)}")
    if os.path.getmtime(WIND) < os.path.getmtime(SCRIPT):
        print("  NOTE: the CSV predates the script on disk -- it may have been written by another version.")

print("\n1. provenance: minimum of the speed columns (must be >= 0)")
ok1 = True
for s in ORDER + ["circumpolar"]:
    c = f"wind_{s}"
    if c in w.columns:
        mn = float(w[c].min())
        flag = "" if mn >= 0 else "   <-- NEGATIVE: not a speed; CSV from another script version"
        ok1 &= mn >= 0
        print(f"   {s:26s} min {mn:+8.3f}{flag}")
    else:
        print(f"   {s:26s} (no speed column)")

# what IS the wind_ column? (not used downstream, but it tells us which extraction wrote the file)
print("\n1b. identity of the wind_ column: correlation with u, v and sqrt(u^2+v^2)")
for s in ORDER + ["circumpolar"]:
    if all(f"{k}_{s}" in w.columns for k in ("wind", "u", "v")):
        u, v, sp = w[f"u_{s}"], w[f"v_{s}"], w[f"wind_{s}"]
        print(f"   {s:26s} r(wind,u) {np.corrcoef(sp, u)[0,1]:+.2f}   r(wind,v) {np.corrcoef(sp, v)[0,1]:+.2f}   "
              f"r(wind,|u,v|) {np.corrcoef(sp, np.hypot(u, v))[0,1]:+.2f}")

print("\n2. geographic components: 1988-2023 mean u (expect clearly > 0, westerlies) and mean v (expect small)")
print("   (columns are SUMS over the mask, so units are arbitrary; only signs and the v/u ratio matter)")
ok2 = True
for s in ORDER + ["circumpolar"]:
    u, v = w.get(f"u_{s}"), w.get(f"v_{s}")
    if u is None or v is None:
        print(f"   {s:26s} (missing u/v)"); continue
    mu, mv, su, sv = u.mean(), v.mean(), u.std(), v.std()
    flag = ""
    if mu <= 0:
        flag += "   <-- mean u not westerly"
    if abs(mv) > 0.6 * abs(mu):
        flag += "   <-- |mean v| comparable to mean u: rotated components?"
    ok2 &= (mu > 0) and (abs(mv) <= 0.6 * abs(mu))
    print(f"   {s:26s} mean u {mu:+10.1f} (sd {su:8.1f})   mean v {mv:+10.1f} (sd {sv:8.1f})   v/u {mv/mu if mu else float('nan'):+.2f}{flag}")

print("\n3. coherence: correlation of daily v anomalies between neighbouring sectors (expect > 0)")
anom = {}
for s in ORDER:
    v = w[f"v_{s}"]
    doy = w[tcol].dt.dayofyear
    anom[s] = v - v.groupby(doy).transform("mean")
ok3 = True
ring = ORDER + [ORDER[0]]
for a, b in zip(ring[:-1], ring[1:]):
    r = float(np.corrcoef(anom[a], anom[b])[0, 1])
    flag = "" if r > 0 else "   <-- anticorrelated neighbours"
    ok3 &= r > 0
    print(f"   {a:26s} - {b:26s} r = {r:+.2f}{flag}")
print("   (Weddell-ABS are neighbours across the Peninsula; a weak or negative value there is plausible)")

print("\nverdict:")
print("   provenance", "OK" if ok1 else "FAIL", "|", "geographic components", "OK" if ok2 else "FAIL",
      "|", "neighbour coherence", "OK" if ok3 else "CHECK")
if ok1 and ok2:
    print("   v is northward wind with one convention in every sector. The Atlantic reversal in\n"
          "   compute_wind_response.py is a property of the ice-edge response, not of the file.")
else:
    print("   regenerate ERA5_winds_daily_sector.csv with the script on disk before reading any sign.")