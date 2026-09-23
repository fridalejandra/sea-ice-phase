#!/usr/bin/env python3
"""
make_table_s2.py -- Table S2: the full 420-correlation scan, in a compact
layout that fits on one landscape page.

Reads   results/ch3/tables/t35_index_scan_raw.csv
        (written by ch3_stats.py, Sect. 3.5 block: 6 series x 2 observed
         targets x 35 seasonal index values = 420 rows;
         columns sector, target, index, n, r, p, q_bh)
Writes  results/ch3/tables/table_s2_scan.md    paste into the supplement
        results/ch3/tables/table_s2_scan.csv   same numbers, plain
        results/ch3/tables/table_s2_scan.docx  only if pandoc is on PATH

Layout: one row per index x season (35 rows), one column per series x
component (12 columns). Each cell is the Pearson r of the detrended annual
series, in bold where p < 0.05 and with a dagger where the Benjamini-
Hochberg q ACROSS THE WHOLE SCAN (column q_bh, as ch3_stats.py computes it)
is < 0.05. The console prints the counts the caption quotes, so you can
check them against the manuscript's "28 of 420, none after FDR".

    cd .../scripts/python/plotting/Ch3/figures
    python make_table_s2.py
    python make_table_s2.py --csv /path/to/t35_index_scan_raw.csv
"""
import os
import sys
import shutil
import subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── where the scan is ────────────────────────────────────────────────────────
if "--csv" in sys.argv:
    CSV = sys.argv[sys.argv.index("--csv") + 1]
    OUT = os.path.dirname(os.path.abspath(CSV))
else:
    from ch3_config import TABLES_DIR as OUT
    CSV = os.path.join(OUT, "t35_index_scan_raw.csv")
if not os.path.exists(CSV):
    sys.exit(f"{CSV} not found; run ch3_stats.py first, or pass --csv")

d = pd.read_csv(CSV)
need = {"sector", "target", "index", "r", "p", "q_bh"}
miss = need - set(d.columns)
if miss:
    sys.exit(f"{CSV} lacks columns {sorted(miss)}; has {sorted(d.columns)}")

# index column is e.g. "Nino34_SON" or "SAM_annual": split at the last "_"
parts = d["index"].astype(str).str.rsplit("_", n=1, expand=True)
d["ix"] = parts[0]
d["season"] = parts[1].fillna("annual")

# ── display names and orders (same as the manuscript) ───────────────────────
INDEX_ORDER = ["Nino34", "SAM", "ASL", "ZW3R", "ZW3G"]
INDEX_NAME = {"Nino34": "Niño3.4", "SAM": "SAM", "ASL": "ASL",
              "ZW3R": "ZW3 (Raphael)", "ZW3G": "ZW3 (Goyal)"}
SEASON_ORDER = ["annual", "DJF", "MAM", "JJA", "SON", "ADV", "RET"]
SEASON_NAME = {"annual": "annual", "DJF": "DJF", "MAM": "MAM", "JJA": "JJA",
               "SON": "SON", "ADV": "Mar–Aug", "RET": "Oct–Jan"}
SECTOR_ORDER = ["Weddell", "ABS", "Ross", "East Antarctica", "King Haakon", "Circumpolar"]
SECTOR_KEY = {"Weddell": "Weddell", "SIE_Weddell": "Weddell",
              "ABS": "ABS", "Amundsen-Bellingshausen": "ABS", "SIE_Amundsen_Bellingshausen": "ABS",
              "Ross": "Ross", "SIE_Ross": "Ross",
              "East Antarctica": "East Antarctica", "SIE_East_Antarctica": "East Antarctica",
              "King Haakon": "King Haakon", "SIE_King_Haakon": "King Haakon",
              "Circumpolar": "Circumpolar", "circumpolar": "Circumpolar", "SIE_circumpolar": "Circumpolar"}
SECTOR_SHORT = {"Weddell": "Wed", "ABS": "A-B", "Ross": "Ross",
                "East Antarctica": "EA", "King Haakon": "KH", "Circumpolar": "Circ"}
# target values as ch3_stats.py writes them (AMP, PH); both spellings accepted
COMP_KEY = {"amplitude_raw_anom": "amp", "amplitude_raw": "amp", "amplitude": "amp",
            "max_doy_raw_anom": "doy", "max_doy_raw": "doy", "max_doy": "doy", "phase": "doy"}
COMP_NAME = {"amp": "amplitude", "doy": "day of max"}

d["sec"] = d["sector"].map(SECTOR_KEY)
d["comp"] = d["target"].map(COMP_KEY)
for col, name in (("sec", "sector"), ("comp", "target"), ("ix", "index"), ("season", "season")):
    ok = {"sec": set(SECTOR_ORDER), "comp": set(COMP_KEY.values()),
          "ix": set(INDEX_ORDER), "season": set(SEASON_ORDER)}[col]
    bad = sorted(set(d[col].dropna().astype(str)) - ok) + (["<NaN>"] if d[col].isna().any() else [])
    if bad:
        print(f"note: {name} values not recognised, dropped: {bad}")
d = d[d["sec"].isin(SECTOR_ORDER) & d["comp"].isin(COMP_NAME) & d["ix"].isin(INDEX_ORDER)
      & d["season"].isin(SEASON_ORDER)].copy()

# ── the counts the caption quotes ────────────────────────────────────────────
n = len(d)
n_p = int((d["p"] < 0.05).sum())
n_q = int((d["q_bh"] < 0.05).sum())
print(f"{n} correlations in the table; {n_p} with p < 0.05 "
      f"({0.05 * n:.0f} expected by chance); {n_q} with q < 0.05")
if n != 420:
    print(f"WARNING: expected 420 rows, got {n} -- check the 'dropped' notes above")
if "n" in d.columns:
    print(f"n per correlation: {sorted(d['n'].unique())}")


def cell(r, p, q, md=True):
    if pd.isna(r):
        return "–"
    s = f"{r:+.2f}".replace("-", "−")   # typographic minus, as in the other tables
    if q < 0.05:
        s += "†"
    if p < 0.05 and md:
        s = f"**{s}**"
    return s


col_keys = [(s, c) for s in SECTOR_ORDER for c in ("amp", "doy")]
col_names = [f"{SECTOR_SHORT[s]} {COMP_NAME[c]}" for s, c in col_keys]

rows_md, rows_csv = [], []
for ix in INDEX_ORDER:
    for se in SEASON_ORDER:
        sub = d[(d["ix"] == ix) & (d["season"] == se)]
        label = f"{INDEX_NAME[ix]}, {SEASON_NAME[se]}"
        md_cells, csv_cells = [], []
        for s, c in col_keys:
            m = sub[(sub["sec"] == s) & (sub["comp"] == c)]
            if len(m) > 1:
                sys.exit(f"more than one row for {ix} {se} {s} {c} -- duplicated scan?")
            if len(m) == 0:
                md_cells.append("–"); csv_cells.append("")
                continue
            r, p, q = float(m.iloc[0]["r"]), float(m.iloc[0]["p"]), float(m.iloc[0]["q_bh"])
            md_cells.append(cell(r, p, q))
            csv_cells.append(cell(r, p, q, md=False))
        rows_md.append([label] + md_cells)
        rows_csv.append([label] + csv_cells)

header = ["Index, season"] + col_names
md = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * len(col_names)]
md += ["| " + " | ".join(r) + " |" for r in rows_md]
md_text = "\n".join(md) + "\n"
md_text += ("\nBold: p < 0.05. †: Benjamini–Hochberg q < 0.05 across the whole scan. "
            f"{n_p} of {n} entries have p < 0.05 ({0.05 * n:.0f} expected by chance); "
            f"{n_q} have q < 0.05.\n")

os.makedirs(OUT, exist_ok=True)
p_md = os.path.join(OUT, "table_s2_scan.md")
p_csv = os.path.join(OUT, "table_s2_scan.csv")
with open(p_md, "w", encoding="utf-8") as f:
    f.write(md_text)
pd.DataFrame(rows_csv, columns=header).to_csv(p_csv, index=False)
print(f"wrote {p_md}\nwrote {p_csv}")

if shutil.which("pandoc"):
    p_docx = os.path.join(OUT, "table_s2_scan.docx")
    r = subprocess.run(["pandoc", p_md, "-o", p_docx], capture_output=True, text=True)
    print(f"wrote {p_docx}" if r.returncode == 0 else f"pandoc failed: {r.stderr.strip()}")
else:
    print("pandoc not on PATH -- paste table_s2_scan.md into the Google Doc, or send it to me")
