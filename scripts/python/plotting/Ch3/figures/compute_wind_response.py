#!/usr/bin/env python3
"""
Wind forcing of the raw APAC anomaly, done properly.

Replaces the daily-lag block of compute_component_comparison.py (lines 349-368).
Five changes from that version, each addressing a specific problem:

 1. BOTH wind components.  Ice drifts 20-40 deg to the LEFT of the wind in the
    Southern Hemisphere (Kottmeier et al. 1992; Eabry et al. 2025), so the
    equatorward drift that moves the edge goes as  u*sin(theta) + v*cos(theta).
    Using v alone discards a zonal term worth 36-84 % of the meridional one.
    Rather than assume theta, we fit u and v jointly and RECOVER it:
        theta_hat = atan2(beta_u, beta_v)
    A sector returning 20-40 deg is confirming the physics. A sector returning
    something far from that is telling you its ice edge is not zonal, which is
    the candidate explanation for the Atlantic/Pacific sign flip.

 2. PARTIAL, not marginal, lags.  Wind is autocorrelated day to day, so
    separate bivariate correlations at lags 0-3 show a decaying tail even when
    the ice has no memory at all (verified by simulation). All lags go in one
    regression.

 3. A DAMPING term.  Including raw(t-1) turns the fit into a discrete
    Ornstein-Uhlenbeck model,  d_raw = a*W - raw/tau + e,  so tau is estimated
    directly and can be compared with the e-folding time from the ACF. If they
    agree, the residual is a damped integrator of wind forcing and tau is a
    physical relaxation time. If not, something else is in there.

 4. CENTRED wind.  d_raw(t) = raw(t) - raw(t-1) is a change over the interval
    [t-1, t], centred at t-0.5, but the ERA5 field is an instantaneous 12 UTC
    snapshot at t. Averaging the snapshots at t-1 and t centres the forcing on
    the interval and stops signal leaking into lag 1.

 5. SEASON.  The raw anomaly's variance swings by an order of magnitude through
    the year (Fig. S3), so a pooled correlation is dominated by the retreat
    months. Everything is reported by season as well as pooled.

Optional block 6 tests Eabry et al.'s preconditioning hypothesis -- that a
loosely packed ice field responds more strongly to wind -- as an interaction
with open-water area. It needs daily sector SIA, which is not in the current
daily CSV; the block is skipped with a message if OWA_CSV is not set.

Usage
-----
    python compute_wind_response.py              # paths come from ch3_config
    python compute_wind_response.py --selftest   # recover known coefficients
    python compute_wind_response.py --raw-wind   # without interval-centring

Paths follow ch3_config, exactly as compute_component_comparison.py does.
Override any of them with the env vars DAILY_CSV, WIND_CSV, OWA_CSV,
YEAR_LO, YEAR_HI.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_config():
    """Locate ch3_config.py even when this script is run from elsewhere
    (e.g. ~/Downloads). Searches the script dir and cwd, walks up from both,
    then looks inside any sea-ice-phase checkout under $HOME."""
    seen = []
    for base in (HERE, os.getcwd()):
        d = base
        for _ in range(6):
            seen.append(d)
            d = os.path.dirname(d)
            if d in ("/", ""):
                break
    for d in seen:
        for sub in ("", "scripts/python/plotting/Ch3/figures",
                    "scripts/python/plotting/Ch3", "scripts/python", "scripts"):
            p = os.path.join(d, sub) if sub else d
            if os.path.exists(os.path.join(p, "ch3_config.py")):
                return p
    home = os.path.expanduser("~")
    for root, dirs, files in os.walk(home):
        dirs[:] = [x for x in dirs if not x.startswith(".")
                   and x not in ("Library", "Applications", "node_modules")]
        if "ch3_config.py" in files:
            return root
        if root.count(os.sep) - home.count(os.sep) > 6:
            dirs[:] = []
    return None


_cfg_dir = _find_config()
sys.path.insert(0, _cfg_dir or HERE)
if _cfg_dir and _cfg_dir != HERE:
    print("using config from %s\n" % _cfg_dir)
try:
    from ch3_config import DAILY_CSV as CFG_DAILY, SECTORS_COMPUTE
    try:
        from ch3_config import RAW_DATA_DIR
    except ImportError:
        RAW_DATA_DIR = os.path.join(HERE, "..", "data", "raw")
    try:
        from ch3_config import TABLES_DIR
    except ImportError:
        TABLES_DIR = HERE
except ImportError:                      # standalone / --selftest
    CFG_DAILY, RAW_DATA_DIR, TABLES_DIR = None, HERE, HERE
    SECTORS_COMPUTE = {}

WIND_SECTOR = {   # daily sector label -> wind column suffix
    "Weddell": "Weddell", "ABS": "Amundsen_Bellingshausen",
    "Amundsen-Bellingshausen": "Amundsen_Bellingshausen", "Ross": "Ross",
    "East Antarctica": "East_Antarctica", "King Haakon": "King_Haakon",
    "Circumpolar": "circumpolar", "circumpolar": "circumpolar",
}

VERSION = "2026-09-21g"                 # printed at startup, so you can tell
                                        # which copy of this file actually ran
MAXLAG = 3
HAC_LAGS = 10                           # Newey-West window for daily autocorrelation
CENTRE = "--raw-wind" not in sys.argv   # centre wind on the tendency interval


def parse_dates(series):
    """Robust date parsing for columns that may hold a timestamp, a date
    string, or the repr of a one-element DatetimeIndex, e.g.
        DatetimeIndex(['1979-01-01 12:00:00'], dtype='datetime64[ns]', freq=None)
    which is what you get when a scalar was never extracted before to_csv."""
    raw = series.astype(str)
    t = pd.to_datetime(raw.str.extract(r"(\d{4}-\d{2}-\d{2})")[0],
                       errors="coerce", format="%Y-%m-%d")
    if t.isna().mean() > 0.5:
        t = pd.to_datetime(raw, errors="coerce")
    return t


SEASONS = {"advance (Mar-Aug)": [3, 4, 5, 6, 7, 8],
           "retreat (Oct-Jan)": [10, 11, 12, 1]}


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
class Fit:
    """OLS with Newey-West (HAC) standard errors. numpy only."""
    def __init__(self, y, X, names, maxlags):
        n, k = X.shape
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ (X.T @ y)
        e = y - X @ b
        # Newey-West meat matrix
        S = (X * e[:, None]).T @ (X * e[:, None])
        for j in range(1, maxlags + 1):
            wj = 1.0 - j / (maxlags + 1.0)
            A = (X[j:] * e[j:, None]).T @ (X[:-j] * e[:-j, None])
            S += wj * (A + A.T)
        V = XtX_inv @ S @ XtX_inv
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        self.params = pd.Series(b, index=names)
        self.bse = pd.Series(se, index=names)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tvalues = pd.Series(np.where(se > 0, b / se, np.nan), index=names)
        self._n = n
        self._V = V
        self._names = list(names)
        sst = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1.0 - float((e ** 2).sum()) / sst if sst > 0 else np.nan
        self._y, self._X = y, X

    def r2_without(self, keys):
        """R^2 of the same fit with the named columns dropped: the share of the
        variance those columns carry once the others are in."""
        keep = [i for i, k in enumerate(self._names) if k not in keys]
        Xk = self._X[:, keep]
        b = np.linalg.pinv(Xk.T @ Xk) @ (Xk.T @ self._y)
        e = self._y - Xk @ b
        sst = float(((self._y - self._y.mean()) ** 2).sum())
        return 1.0 - float((e ** 2).sum()) / sst if sst > 0 else np.nan

    def combo(self, keys):
        """Estimate and HAC standard error of the SUM of several coefficients.
        With autocorrelated predictors the individual lag coefficients are
        poorly identified while their sum -- the cumulative response to a
        sustained anomaly -- is stable, so this is the number to read."""
        c = np.zeros(len(self._names))
        for k in keys:
            if k in self._names:
                c[self._names.index(k)] = 1.0
        est = float(c @ self.params.to_numpy())
        var = float(c @ self._V @ c)
        se = np.sqrt(var) if var > 0 else np.nan
        return est, se, (est / se if se and np.isfinite(se) else np.nan)


def fit(d, use_lags=True, use_damping=True, label=""):
    """d_raw on wind components (+lags, +damping), HAC errors."""
    cols = {}
    lags = range(MAXLAG + 1) if use_lags else [0]
    for L in lags:
        cols[f"u{L}"] = d["u_c"].shift(L)
        cols[f"v{L}"] = d["v_c"].shift(L)
    if use_damping:
        cols["raw_lag"] = d["raw"].shift(1)
    X = pd.DataFrame(cols, index=d.index)
    y = d["d_raw"]
    ok = X.notna().all(axis=1) & y.notna()
    if ok.sum() < 100:
        return None
    Xm = np.column_stack([np.ones(int(ok.sum())), X[ok].to_numpy(float)])
    m = Fit(y[ok].to_numpy(float), Xm, ["const"] + list(X.columns), HAC_LAGS)
    m._label = label
    return m


def deflection(m):
    """Effective ice-drift deflection angle implied by the lag-0 coefficients."""
    bu, bv = m.params.get("u0", np.nan), m.params.get("v0", np.nan)
    if not np.isfinite(bu) or not np.isfinite(bv):
        return np.nan, np.nan
    ang = np.degrees(np.arctan2(bu, bv))
    mag = float(np.hypot(bu, bv))
    return ang, mag


def tau_from(m):
    """Relaxation time implied by the damping coefficient, in days."""
    b = m.params.get("raw_lag", np.nan)
    if not np.isfinite(b) or b >= 0:
        return np.nan
    return float(-1.0 / b)


def report(m, extra=""):
    if m is None:
        print("      (too few observations)")
        return
    ang, mag = deflection(m)
    bits = [f"n={m._n}"]
    for k in ("u0", "v0"):
        if k in m.params:
            bits.append(f"{k} {m.params[k]:+.5f} (t {m.tvalues[k]:+.1f})")
    if np.isfinite(ang):
        bits.append(f"deflection {ang:+.0f} deg")
    if "raw_lag" in m.params:
        t = tau_from(m)
        bits.append(f"tau {t:.1f} d" if np.isfinite(t) else "tau n/a")
    bits.append(f"R2 {m.r2:.3f}")
    wind_keys = [k for k in m._names if k[0] in "uv" and k[1:].isdigit()]
    if wind_keys:
        # variance the wind terms carry, with the damping term (if any) kept
        bits.append(f"wind share {m.r2 - m.r2_without(wind_keys):.3f}")
    print("      " + "   ".join(bits) + extra)
    if "u1" in m.params:
        lag_line = []
        for L in range(MAXLAG + 1):
            bu = m.params.get(f"u{L}", np.nan); bv = m.params.get(f"v{L}", np.nan)
            tu = m.tvalues.get(f"u{L}", np.nan); tv = m.tvalues.get(f"v{L}", np.nan)
            lag_line.append(f"L{L}: u {bu:+.5f}({tu:+.0f}) v {bv:+.5f}({tv:+.0f})")
        print("        " + " | ".join(lag_line))
        # alternating signs across lags 1-3 mean the timing is still misspecified,
        # not that the ice has multi-day memory
        cu = m.combo([f"u{L}" for L in range(MAXLAG + 1)])
        cv = m.combo([f"v{L}" for L in range(MAXLAG + 1)])
        cang = np.degrees(np.arctan2(cu[0], cv[0]))
        print("        CUMULATIVE over lags 0-3:  u %+.5f (t %+.1f)   v %+.5f (t %+.1f)"
              "   deflection %+.0f deg" % (cu[0], cu[2], cv[0], cv[2], cang))
        sg = [np.sign(m.params.get(f"v{L}", 0.0)) for L in (1, 2, 3)]
        tt = [abs(m.tvalues.get(f"v{L}", 0.0)) for L in (1, 2, 3)]
        if len(set(sg)) > 1 and sg[0] != sg[1] and sg[1] != sg[2] and min(tt) > 2:
            print("        ^ lag signs alternate and are significant: the forcing is"
                  " still mistimed relative to the tendency.")
            print("          Compare with --raw-wind before reading these as memory.")


# ─────────────────────────────────────────────────────────────────────────────
# self-test: can the machinery recover a known answer?
# ─────────────────────────────────────────────────────────────────────────────
def selftest():
    print("compute_wind_response.py  version %s\n" % VERSION)
    rng = np.random.default_rng(1)
    n = 13000
    TRUE_THETA, TRUE_TAU, TRUE_GAIN = 30.0, 15.0, 0.05
    phi = 0.75
    u = np.zeros(n); v = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t-1] + np.sqrt(1 - phi**2) * rng.standard_normal()
        v[t] = phi * v[t-1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    th = np.radians(TRUE_THETA)
    W = u * np.sin(th) + v * np.cos(th)          # equatorward drift forcing
    raw = np.zeros(n)
    for t in range(1, n):
        raw[t] = raw[t-1] + TRUE_GAIN * W[t] - raw[t-1] / TRUE_TAU \
                 + 0.02 * rng.standard_normal()
    d = pd.DataFrame({"u_c": u, "v_c": v, "raw": raw})
    d["d_raw"] = d["raw"].diff()

    print("SELF-TEST  truth: deflection %.0f deg, tau %.0f d, |gain| %.3f\n"
          % (TRUE_THETA, TRUE_TAU, TRUE_GAIN))
    print("  their model (v only, no lags, no damping):")
    m0 = fit(d.assign(u_c=0.0), use_lags=False, use_damping=False)
    report(m0)
    print("  + both components:")
    report(fit(d, use_lags=False, use_damping=False))
    print("  + lags:")
    report(fit(d, use_lags=True, use_damping=False))
    print("  + damping   <-- full model, should recover the truth:")
    report(fit(d, use_lags=True, use_damping=True))
    print("\n  Marginal correlations the old way (note the spurious decaying tail):")
    dd = d.dropna()
    print("   ", "  ".join(
        "lag%d %+0.2f" % (L, dd["v_c"].shift(L).corr(dd["d_raw"])) for L in range(4)))


# ─────────────────────────────────────────────────────────────────────────────
# real data
# ─────────────────────────────────────────────────────────────────────────────
def main():
    DAILY = os.environ.get("DAILY_CSV", CFG_DAILY)
    WIND = os.environ.get("WIND_CSV",
                          os.path.join(RAW_DATA_DIR, "ERA5_winds_daily_sector.csv"))
    missing = [("DAILY_CSV", DAILY), ("WIND_CSV", WIND)]
    missing = [(n, p) for n, p in missing if not (p and os.path.exists(p))]
    if missing:
        print("Could not find:")
        for n, p in missing:
            print("   %-10s %s" % (n, p or "(unset — is ch3_config importable?)"))
            d = os.path.dirname(p) if p else None
            if d and os.path.isdir(d):
                near = [f for f in sorted(os.listdir(d)) if f.endswith(".csv")][:12]
                print("              %s contains: %s" % (d, ", ".join(near) or "(no CSVs)"))
            elif d:
                print("              %s does not exist" % d)
        print("\nEither put the file there, or run with an override, e.g.")
        print("   WIND_CSV=/path/to/ERA5_winds_daily_sector.csv python %s"
              % os.path.basename(__file__))
        print("\nThe wind CSV is the output of compute_ERA5_winds_daily_sector.py")
        print("and must have signed u_<sector> and v_<sector> columns, not just wind_.")
        sys.exit(1)

    lo = int(os.environ.get("YEAR_LO", 1988))
    hi = int(os.environ.get("YEAR_HI", 2023))

    daily = pd.read_csv(DAILY)
    dcol = "Date" if "Date" in daily.columns else daily.columns[0]
    daily["Date"] = parse_dates(daily[dcol])
    if daily["Date"].isna().all():
        sys.exit("could not parse the Date column of %s; first values: %s"
                 % (DAILY, daily[dcol].astype(str).head(3).tolist()))
    daily = daily.dropna(subset=["Date"])
    if "period" in daily.columns:
        daily = daily[daily["period"] == "FULL"]
    w = pd.read_csv(WIND)
    # ERA5 variable names (u10_/v10_) and the short form (u_/v_) both occur,
    # depending on which extraction script wrote the file. Normalise to u_/v_,
    # exactly as compute_component_comparison.py does at its line 293.
    w = w.rename(columns={c: c.replace("u10_", "u_").replace("v10_", "v_")
                          for c in w.columns})
    # The time column may hold the repr of a one-element DatetimeIndex rather
    # than a timestamp -- "DatetimeIndex(['1979-01-01 12:00:00'], dtype=...)" --
    # depending on which extraction script wrote the file. Pull the first
    # YYYY-MM-DD out of whatever the string is, as compute_component_comparison.py
    # does at its line 283.
    tcol = "time" if "time" in w.columns else w.columns[0]
    t = parse_dates(w[tcol])
    if t.isna().mean() > 0.5:
        sys.exit("could not parse the wind time column; first values: %s"
                 % w[tcol].astype(str).head(3).tolist())
    w["Date"] = t.dt.normalize()
    w = w.dropna(subset=["Date"])

    suffixes = sorted({c[2:] for c in w.columns if c.startswith("v_")})
    if not suffixes:
        sys.exit("wind CSV has no v_<sector> or v10_<sector> columns — it is the\n"
                 "speed-only file. Rerun the extraction for signed u, v.\n"
                 "columns seen: %s" % ", ".join(list(w.columns)[:8]))

    # map each wind suffix to the sector label used in the daily file
    labels = {str(x): str(x) for x in daily["sector"].dropna().unique()}
    low = {k.lower(): v for k, v in labels.items()}
    def daily_rows(suf):
        cands = [k for k, v in WIND_SECTOR.items() if v == suf] + [suf, "SIE_" + suf]
        cands += [c.replace("_", " ") for c in list(cands)]
        for c in cands:
            if c.lower() in low:
                return daily[daily["sector"].astype(str) == low[c.lower()]]
        return daily.iloc[0:0]

    print("compute_wind_response.py  version %s\n" % VERSION)
    print("daily %s\nwind  %s\nyears %d-%d   HAC maxlags %d   wind %s\n"
          % (DAILY, WIND, lo, hi, HAC_LAGS,
             "centred on [t-1,t]" if CENTRE else "uncentred (12 UTC snapshot)"))

    owa = None
    OWA_CSV = os.environ.get("OWA_CSV")
    if OWA_CSV and os.path.exists(OWA_CSV):
        owa = pd.read_csv(OWA_CSV)
        owa["Date"] = parse_dates(owa[owa.columns[0] if "Date" not in owa.columns else "Date"])
        print("open-water area: %s\n" % OWA_CSV)

    rows = []
    for suf in suffixes:
        sec = daily_rows(suf)
        if sec.empty:
            print("== %-26s no matching sector in the daily file (has: %s), skipped"
                  % (suf, ", ".join(sorted(labels)[:6])))
            continue

        ws = w[["Date", f"u_{suf}", f"v_{suf}"]].rename(
            columns={f"u_{suf}": "u", f"v_{suf}": "v"})
        d = (sec[["Date", "residual_apac"]].rename(columns={"residual_apac": "raw"})
             .merge(ws, on="Date", how="outer").sort_values("Date"))

        # complete daily index, so .diff() can never span a gap
        full = pd.DataFrame({"Date": pd.date_range(d["Date"].min(), d["Date"].max(), freq="D")})
        d = full.merge(d, on="Date", how="left")
        d["Year"] = d["Date"].dt.year
        d["month"] = d["Date"].dt.month
        d["doy"] = d["Date"].dt.dayofyear

        # centre the wind on the tendency interval [t-1, t]. --raw-wind turns
        # this off, to check whether centring helps: if the centring assumption
        # is right, centred lags 1-3 collapse toward zero and uncentred ones do
        # not. Alternating signs across lags mean the timing is still off.
        if CENTRE:
            d["u_c"] = (d["u"] + d["u"].shift(1)) / 2.0
            d["v_c"] = (d["v"] + d["v"].shift(1)) / 2.0
        else:
            d["u_c"], d["v_c"] = d["u"], d["v"]
        for c in ("u_c", "v_c"):
            d[c] = d[c] - d.groupby("doy")[c].transform("mean")

        # The wind columns are a SUM over the sector mask, so their scale is
        # arbitrary and raw coefficients come out at 1e-5. Standardise each
        # component to unit variance: coefficients then read as
        # "10^6 km^2 per day per standard deviation of wind", comparable
        # across sectors. The deflection angle and tau are unaffected.
        for c in ("u_c", "v_c"):
            sd = d[c].std()
            if sd and np.isfinite(sd):
                d[c] = d[c] / sd

        d["d_raw"] = d["raw"].diff()
        d.loc[d["raw"].shift(1).isna(), "d_raw"] = np.nan   # never diff across a gap
        d = d[(d["Year"] >= lo) & (d["Year"] <= hi)]

        print("== %s" % suf)
        print("    their model (v only, no lags, no damping):")
        report(fit(d.assign(u_c=0.0), use_lags=False, use_damping=False))
        print("    full model (u+v, lags 0-3, damping):")
        m = fit(d, use_lags=True, use_damping=True)
        report(m)

        for sname, months in SEASONS.items():
            ds = d[d["month"].isin(months)]
            ms = fit(ds, use_lags=True, use_damping=True, label=sname)
            print("    %-20s" % sname)
            report(ms)
            if ms is not None:
                a, _ = deflection(ms)
                cu = ms.combo([f"u{L}" for L in range(MAXLAG + 1)])
                cv = ms.combo([f"v{L}" for L in range(MAXLAG + 1)])
                rows.append({"sector": suf, "season": sname, "n": ms._n,
                             "beta_u": ms.params.get("u0"), "beta_v": ms.params.get("v0"),
                             "deflection_deg": a, "tau_d": tau_from(ms),
                             "cum_u": cu[0], "cum_u_t": cu[2],
                             "cum_v": cv[0], "cum_v_t": cv[2],
                             "cum_deflection_deg": np.degrees(np.arctan2(cu[0], cv[0]))})

        # ── optional: Eabry preconditioning, wind gain vs pack state ─────────
        if owa is not None and f"owa_{suf}" in owa.columns:
            o = owa[["Date", f"owa_{suf}"]].rename(columns={f"owa_{suf}": "owa"})
            do = d.merge(o, on="Date", how="left")
            do["owa_a"] = do["owa"] - do.groupby("doy")["owa"].transform("mean")
            do["owa_a"] = do["owa_a"] / do["owa_a"].std()
            th = np.radians(30.0)
            do["W"] = do["u_c"] * np.sin(th) + do["v_c"] * np.cos(th)
            do["W_x_owa"] = do["W"] * do["owa_a"]
            X = do[["W", "W_x_owa", "owa_a"]].copy()
            X["raw_lag"] = do["raw"].shift(1)
            ok = X.notna().all(axis=1) & do["d_raw"].notna()
            if ok.sum() > 200:
                Xm = np.column_stack([np.ones(int(ok.sum())), X[ok].to_numpy(float)])
                mi = Fit(do["d_raw"][ok].to_numpy(float), Xm,
                         ["const"] + list(X.columns), HAC_LAGS)
                print("    preconditioning (wind x open-water area):")
                print("      W %+.4f (t %+.1f)   W*OWA %+.4f (t %+.1f)  <- positive means"
                      " a loose pack responds more"
                      % (mi.params["W"], mi.tvalues["W"],
                         mi.params["W_x_owa"], mi.tvalues["W_x_owa"]))
        print()

    if rows:
        out = pd.DataFrame(rows)
        p = os.path.join(TABLES_DIR if os.path.isdir(TABLES_DIR) else ".", "wind_response_by_sector.csv")
        out.to_csv(p, index=False)
        print("wrote", p)
        print("\nDeflection angle by sector and season (expect 20-40 deg where the")
        print("ice edge is zonal; a large departure points at edge geometry):")
        print(out.pivot_table(index="sector", columns="season",
                              values="deflection_deg").round(0).to_string())
        print("\nCUMULATIVE response to a sustained wind anomaly (sum over lags 0-3).")
        print("This is the number to quote: individual lag coefficients are poorly")
        print("identified when the predictor is autocorrelated, but their sum is not.")
        print(out.pivot_table(index="sector", columns="season",
                              values="cum_v").round(5).to_string())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()