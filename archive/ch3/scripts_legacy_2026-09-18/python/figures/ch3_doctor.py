"""
ch3_doctor.py — one-shot environment check. Run from your figures directory:

    python ch3_doctor.py

Prints what resolves, what doesn't, and why. Every check is independently
wrapped, so one failure doesn't hide the rest. Change nothing, write nothing.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def ok(msg):   print(f"  OK    {msg}")
def bad(msg):  print(f"  FAIL  {msg}")
def warn(msg): print(f"  WARN  {msg}")

print("=" * 72)
print("ch3_doctor")
print("=" * 72)
print(f"python   : {sys.version.split()[0]}")
print(f"cwd      : {os.getcwd()}")
print(f"script   : {HERE}")

# ── 1. config ───────────────────────────────────────────────────────────────
print("\n[1] ch3_config")
cfg = None
try:
    import ch3_config as cfg
    ok(f"imported")
    print(f"        ROOT       = {cfg.ROOT}")
    print(f"        DATA_DIR   = {cfg.DATA_DIR}")
    print(f"        OUTPUT_DIR = {cfg.OUTPUT_DIR}")
    print(f"        TABLES_DIR = {cfg.TABLES_DIR}")
    print(f"        YEAR_MIN/MAX = {cfg.YEAR_MIN} / {cfg.YEAR_MAX}")
    print(f"        BREAK_YEAR   = {cfg.BREAK_YEAR}")
except Exception:
    bad("could not import ch3_config")
    traceback.print_exc()
    print("\nStopping: nothing else can work without the config.")
    sys.exit(1)

# ── 2. the CSVs ─────────────────────────────────────────────────────────────
print("\n[2] data files")
import datetime
for name in ("DAILY_CSV", "ANNUAL_CSV", "RMSE_CSV"):
    p = getattr(cfg, name, None)
    if p is None:
        bad(f"{name} not defined in ch3_config"); continue
    if os.path.exists(p):
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(p))
        ok(f"{name:11s} {p}")
        print(f"        {os.path.getsize(p)/1e6:.1f} MB, modified {mt:%Y-%m-%d %H:%M}")
    else:
        bad(f"{name:11s} MISSING: {p}")

# also flag stale _E siblings, which several scripts still fall back to
try:
    for f in sorted(os.listdir(cfg.DATA_DIR)):
        if f.endswith("_E.csv"):
            warn(f"stale file present: {os.path.join(cfg.DATA_DIR, f)}")
except Exception:
    pass

# ── 3. daily contents ───────────────────────────────────────────────────────
print("\n[3] daily_fitted contents")
try:
    import pandas as pd
    d = pd.read_csv(cfg.DAILY_CSV)
    ok(f"{len(d):,} rows, {len(d.columns)} columns")

    if "period" in d.columns:
        counts = d["period"].value_counts().to_dict()
        warn(f"period column present: {counts}")
        print("        -> any script using plain pd.read_csv on this file is")
        print("           double-counting the overlapping years. Filter to FULL.")
        d = d[d["period"] == "FULL"]
        print(f"        after period=='FULL': {len(d):,} rows")
    else:
        ok("no period column (single-period file)")

    need = ["Date", "Year", "DOY", "sector", "Extent", "anomaly_from_iac",
            "iac_notrend", "trend_component", "amplitude_component",
            "phase_component", "residual_apac"]
    miss = [c for c in need if c not in d.columns]
    (ok("all expected columns present") if not miss
     else bad(f"missing columns: {miss}"))
    print(f"        columns: {sorted(d.columns)}")

    if "Year" in d.columns:
        print(f"        years  : {int(d.Year.min())} to {int(d.Year.max())}")
        if int(d.Year.max()) > cfg.YEAR_MAX:
            warn(f"data runs to {int(d.Year.max())} but YEAR_MAX is {cfg.YEAR_MAX};"
                 f" scripts filtering on YEAR_MAX will silently drop the tail")
    if "sector" in d.columns:
        print(f"        sectors: {sorted(d.sector.unique())}")

    comps = ["trend_component", "amplitude_component", "phase_component", "residual_apac"]
    if not [c for c in comps + ["anomaly_from_iac"] if c not in d.columns]:
        err = (d[comps].sum(axis=1) - d["anomaly_from_iac"]).abs().max()
        (ok(f"decomposition sums (max abs error {err:.2e})") if err < 1e-6
         else bad(f"decomposition does NOT sum (max abs error {err:.2e})"))
except Exception:
    bad("could not read/inspect the daily file")
    traceback.print_exc()

# ── 4. annual contents ──────────────────────────────────────────────────────
print("\n[4] annual_params contents")
try:
    import pandas as pd
    a = pd.read_csv(cfg.ANNUAL_CSV)
    ok(f"{len(a):,} rows")
    if "period" in a.columns:
        warn(f"period column present: {a['period'].value_counts().to_dict()}")
        a = a[a["period"] == "FULL"]
    need = ["Year", "sector", "amplitude_raw_anom", "amplitude_anom",
            "min_doy_raw_anom", "min_doy_anom",
            "max_doy_raw_anom", "max_doy_anom"]
    miss = [c for c in need if c not in a.columns]
    (ok("all expected columns present") if not miss
     else bad(f"missing columns: {miss}"))
    print(f"        columns: {sorted(a.columns)}")
    if "Year" in a.columns:
        print(f"        years  : {int(a.Year.min())} to {int(a.Year.max())}"
              f"  (n={a.Year.nunique()})")
except Exception:
    bad("could not read/inspect the annual file")
    traceback.print_exc()

# ── 5. helper modules ───────────────────────────────────────────────────────
print("\n[5] helper modules")
for mod in ("ch3_data", "ch3_plot"):
    try:
        __import__(mod); ok(f"{mod} imports")
    except Exception:
        bad(f"{mod} failed to import"); traceback.print_exc()

try:
    import ch3_data as D
    for period in ("FULL", "HR2018"):
        try:
            x = D.load_daily(period=period)
            ok(f"load_daily(period={period!r}) -> {len(x):,} rows")
        except SystemExit as e:
            bad(f"load_daily(period={period!r}) refused: {e}")
        except Exception as e:
            bad(f"load_daily(period={period!r}) errored: {type(e).__name__}: {e}")
except Exception:
    pass

# ── 6. scripts that bypass the loaders ──────────────────────────────────────
print("\n[6] scripts reading the CSVs directly (period bug risk)")
try:
    import re
    pat = re.compile(r"read_csv\s*\(\s*(DAILY_CSV|ANNUAL_CSV)")
    hits = []
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".py") or f == os.path.basename(__file__):
            continue
        try:
            src = open(os.path.join(HERE, f), encoding="utf-8").read()
        except Exception:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if pat.search(line):
                hits.append(f"{f}:{i}")
    if hits:
        for h in hits:
            warn(f"direct read: {h}")
        print("        -> these bypass ch3_data's period filter")
    else:
        ok("none found in this directory")
except Exception:
    pass

print("\n" + "=" * 72)
print("done")
print("=" * 72)
