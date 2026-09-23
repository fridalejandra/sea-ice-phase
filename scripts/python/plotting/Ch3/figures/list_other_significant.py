#!/usr/bin/env python3
"""
list_other_significant.py -- which index-sector-season-component combinations
are significant but are NOT among the seven Table 3 relationships?

Finds the full scan table in TABLES_DIR (any CSV with sector, index_base,
season and r_/p_ columns and more than seven rows, e.g. the one
compute_component_comparison.py uses for the "50 of 210" counts), or takes a
path as the first argument.

For each component it prints
  - how many combinations are significant (p < 0.05) and how many chance gives
  - every significant one outside Table 3, with
      n_seasons  how many seasons of the same sector x index x component are
                 significant with the SAME sign (1 = a lone season, more likely chance)
Writes results/ch3/tables/tS_other_significant.csv
Run from the figures directory:  python list_other_significant.py [scan.csv]
"""
import glob
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ch3_config import TABLES_DIR

TABLE3 = {("Nino34", "SON", "ABS"), ("Nino34", "annual", "King Haakon"),
          ("SAM", "RET", "East Antarctica"), ("SAM", "JJA", "Weddell"),
          ("ASL", "annual", "Ross"), ("ZW3R", "SON", "King Haakon"),
          ("ZW3R", "annual", "ABS")}
T3COMP = {("Nino34", "SON", "ABS"): "amplitude", ("Nino34", "annual", "King Haakon"): "amplitude",
          ("SAM", "RET", "East Antarctica"): "max_doy", ("SAM", "JJA", "Weddell"): "amplitude",
          ("ASL", "annual", "Ross"): "amplitude", ("ZW3R", "SON", "King Haakon"): "max_doy",
          ("ZW3R", "annual", "ABS"): "amplitude"}
KEY = {"SIE_Weddell": "Weddell", "SIE_Amundsen_Bellingshausen": "ABS", "Amundsen-Bellingshausen": "ABS",
       "Amundsen_Bellingshausen": "ABS", "SIE_Ross": "Ross", "SIE_East_Antarctica": "East Antarctica",
       "East_Antarctica": "East Antarctica", "SIE_King_Haakon": "King Haakon",
       "King_Haakon": "King Haakon", "SIE_circumpolar": "circumpolar", "Circumpolar": "circumpolar"}
COMPS = ["sie_anom", "amplitude", "max_doy", "min_doy", "raw_anom"]
NAME = {"sie_anom": "departure", "amplitude": "amplitude", "max_doy": "phase (day of max.)",
        "min_doy": "phase (day of min.)", "raw_anom": "raw anomaly"}


def find_scan():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for p in sorted(glob.glob(os.path.join(TABLES_DIR, "*.csv"))):
        try:
            h = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if {"sector", "index_base", "season"} <= set(h.columns) and any(c.startswith("p_") for c in h.columns):
            n = sum(1 for _ in open(p)) - 1
            if n > 7 and "wind" not in os.path.basename(p):
                print(f"using {p} ({n} rows)")
                return p
    sys.exit("no scan table found; pass its path as the first argument")


t = pd.read_csv(find_scan())
t["sector"] = t["sector"].map(lambda s: KEY.get(s, s))
out = []
for c in COMPS:
    if f"r_{c}" not in t.columns:
        continue
    g = t[["sector", "index_base", "season", f"r_{c}", f"p_{c}"]].dropna()
    g = g.rename(columns={f"r_{c}": "r", f"p_{c}": "p"})
    sig = g[g.p < 0.05].copy()
    print(f"\n== {NAME[c]}: {len(sig)} of {len(g)} significant, about {0.05 * len(g):.0f} expected by chance")
    if sig.empty:
        continue
    sig["sign"] = sig.r > 0
    sig["n_seasons"] = sig.groupby(["sector", "index_base", "sign"])["r"].transform("size")
    sig["in_table3"] = [(b, s, k) in TABLE3 and T3COMP[(b, s, k)] == c
                        for b, s, k in zip(sig.index_base, sig.season, sig.sector)]
    other = sig[~sig.in_table3].sort_values(["n_seasons", "p"], ascending=[False, True])
    for _, r in other.iterrows():
        print(f"   {r.sector:16s} {r.index_base:7s} {r.season:7s} r = {r.r:+.2f}  p = {r.p:.3f}"
              f"   same sign in {r.n_seasons} season(s)")
    out.append(other.assign(component=NAME[c]))
if out:
    res = pd.concat(out).drop(columns=["sign", "in_table3"])
    res.to_csv(os.path.join(TABLES_DIR, "tS_other_significant.csv"), index=False)
    print(f"\nwrote {os.path.join(TABLES_DIR, 'tS_other_significant.csv')}")
print("\nReading it: a combination significant in several seasons with the same sign is worth a sentence;"
      "\na lone season is what the chance count predicts and is best left in the supplement.")
