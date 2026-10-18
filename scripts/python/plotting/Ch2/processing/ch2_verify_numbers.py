#!/usr/bin/env python
"""
ch2_verify_numbers.py -- recompute the numbers quoted in Chapter 2 from the v3 anomaly
files and print each next to the value in the text.

Covers: active pixels, method differences (4.2), post-2016 shifts and season changes (4.3),
sign classes, disagreement fractions, and within-period trends (4.4).
Not covered (come straight from dedicated scripts): summer survival, consolidation time,
zone non-detection, variance ratios, threshold crossings, sensitivity tests.
"""
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PRE, POST = (1979, 2015), (2016, 2024)
AUG15_DOY = 227


def _var(path, names):
    with xr.open_dataset(path, decode_times=False) as ds:
        for n in names:
            if n in ds:
                return ds[n].load()
    raise KeyError(f"{names} not in {path}")


def load(phase, method):
    t = TAGS[(phase, method)]
    cs = [f"{t}_clim_dsa", f"{t}_clim"] if phase == "MS" else [f"{t}_clim"]
    an = [f"{t}_anom_dsa", f"{t}_anom"] if phase == "MS" else [f"{t}_anom"]
    clim = _var(ANOM / f"{t}_climatology.nc", cs).values
    anom = _var(ANOM / f"{t}_anomalies.nc", an)
    return clim, clim[None] + anom.values, anom["year"].values


def slope(y, x):
    """OLS slope along axis 0, ignoring NaNs (days per year)."""
    m = np.isfinite(y)
    X = np.where(m, x[:, None, None], np.nan)
    xm, ym = np.nanmean(X, 0), np.nanmean(y, 0)
    num = np.nansum((X - xm) * (y - ym), 0)
    den = np.nansum((X - xm) ** 2, 0)
    out = num / den
    out[m.sum(0) < 3] = np.nan
    return out


def line(label, value, claimed):
    print(f"  {label:<58s} {value:>10s}   text: {claimed}")


def main():
    with xr.open_dataset(SECT) as m:
        ocean = m["valid_ocean"].astype(bool).values
        sid = m["sector_id"].values

    C, D, Y = {}, {}, None
    for k in TAGS:
        C[k], D[k], Y = load(*k)
    ipre = (Y >= PRE[0]) & (Y <= PRE[1])
    ipost = (Y >= POST[0]) & (Y <= POST[1])
    act = {ph: ocean & np.all([np.isfinite(D[(ph, m_)]).mean(0) >= 0.8 for m_ in ("static", "dynamic")], 0)
           for ph in ("FS", "MS")}

    print("\n=== Active pixels (Methods) ===")
    line("FS active pixels", f"{act['FS'].sum()}", "23,845")
    line("MS active pixels", f"{act['MS'].sum()}", "23,503")

    print("\n=== 4.2 Method differences (climatology, dynamic - static) ===")
    for ph, sign, claim in (("FS", 1, "94% positive"), ("MS", -1, "87% negative")):
        d = (C[(ph, "dynamic")] - C[(ph, "static")])[act[ph]]
        d = d[np.isfinite(d)]
        line(f"{ph}: share of active pixels with that sign", f"{100*np.mean(sign*d > 0):.1f}%", claim)
        line(f"{ph}: mean difference over active pixels", f"{np.mean(d):+.1f}",
             "+9.5 (FS) / -8 (MS, 8.1)")
    print("  sector median of paired differences (text FS: KHV 4.9 ... EA 11.5; MS: ABS 5.3 ... RS 12.4)")
    for ph in ("FS", "MS"):
        vals = []
        for s, lab in SECTORS.items():
            sel = act[ph] & (sid == s)
            d = (C[(ph, "dynamic")] - C[(ph, "static")])[sel]
            vals.append(f"{lab} {np.nanmedian(d):+.1f}")
        print(f"    {ph}: " + ", ".join(vals))

    print("  inner / outer third of each sector (text: FS +16.6 / +2.8; MS -14.0 / -1.0)")
    for ph in ("FS", "MS"):
        inner, outer = [], []
        for s in SECTORS:
            sel = act[ph] & (sid == s)
            st = C[(ph, "static")][sel]
            d = (C[(ph, "dynamic")] - C[(ph, "static")])[sel]
            lo, hi = np.nanpercentile(st, [100 / 3, 200 / 3])
            early, late = d[st <= lo], d[st >= hi]
            i, o = (early, late) if ph == "FS" else (late, early)
            inner.append(i), outer.append(o)
        print(f"    {ph}: inner {np.nanmean(np.concatenate(inner)):+.1f}, outer {np.nanmean(np.concatenate(outer)):+.1f}")

    print("  sign of sector-mean yearly difference, and its year-to-year SD (text: same sign every year "
          "except ABS MS 93%; SD mostly 1-2 d)")
    for ph, sign in (("FS", 1), ("MS", -1)):
        out = []
        for s, lab in SECTORS.items():
            sel = act[ph] & (sid == s)
            yearly = np.nanmean((D[(ph, "dynamic")] - D[(ph, "static")])[:, sel], 1)
            yearly = yearly[np.isfinite(yearly)]
            out.append(f"{lab} {100*np.mean(sign*yearly > 0):.0f}%/SD {np.std(yearly, ddof=1):.1f}")
        print(f"    {ph}: " + ", ".join(out))

    print("\n=== 4.3 Post-2016 shift (2016-2024 minus 1979-2015), active pixels ===")
    shift = {k: np.nanmean(D[k][ipost], 0) - np.nanmean(D[k][ipre], 0) for k in D}
    for m_ in ("static", "dynamic"):
        line(f"FS shift, all active ({m_})", f"{np.nanmean(shift[('FS', m_)][act['FS']]):+.1f}", "about +8")
        line(f"MS shift, all active ({m_})", f"{np.nanmean(shift[('MS', m_)][act['MS']]):+.1f}", "about -7")
    both = act["FS"] & act["MS"]
    season = {}
    for m_ in ("static", "dynamic"):
        dur = (D[("MS", m_)] + AUG15_DOY) - D[("FS", m_)]
        season[m_] = np.nanmean(dur[ipost], 0) - np.nanmean(dur[ipre], 0)
        line(f"season-length change, pixels active for FS and MS ({m_})",
             f"{np.nanmean(season[m_][both]):+.1f}", "-14.8 (static) / -15.3 (dynamic)")
    print("  by sector: FS shift, MS shift, season change (text: WS FS +9.5, MS -12.5/-14.5, season -22/-24;")
    print("  KHV -17; EA -13; RS -11; ABS FS +4/+5, MS ~+1, season -2/-3)")
    tot = {m_: np.nansum(season[m_][both]) for m_ in season}
    for s, lab in SECTORS.items():
        parts = []
        for m_ in ("static", "dynamic"):
            fs = np.nanmean(shift[("FS", m_)][act["FS"] & (sid == s)])
            ms = np.nanmean(shift[("MS", m_)][act["MS"] & (sid == s)])
            se = season[m_][both & (sid == s)]
            share = 100 * np.nansum(se) / tot[m_]
            parts.append(f"{m_[:3]}: FS {fs:+.1f} MS {ms:+.1f} season {np.nanmean(se):+.1f} (share {share:.0f}%)")
        print(f"    {lab:4s} " + " | ".join(parts))
    print("  (text: KHV share ~40%, WS ~25%)")

    print("\n=== 4.3 Sign classes (threshold 0) and pixel-level uncertainty ===")
    for ph, cl in (("FS", "later 80 / earlier 10 / opposite 10"), ("MS", "earlier 75 / later 13 / opposite 13")):
        a = act[ph]
        d, s_ = shift[(ph, "dynamic")][a], shift[(ph, "static")][a]
        ok = np.isfinite(d) & np.isfinite(s_)
        d, s_ = d[ok], s_[ok]
        later, earlier = np.mean((d > 0) & (s_ > 0)), np.mean((d < 0) & (s_ < 0))
        opp = np.mean(((d > 0) & (s_ < 0)) | ((d < 0) & (s_ > 0)))
        opp7 = np.mean(((d > 7) & (s_ < -7)) | ((d < -7) & (s_ > 7)))
        line(f"{ph}: both later / both earlier / opposite (%)",
             f"{100*later:.1f}/{100*earlier:.1f}/{100*opp:.1f}", cl)
        line(f"{ph}: opposite, counting only shifts > 7 days (%)", f"{100*opp7:.1f}", "0.2 (FS) / 0.6 (MS)")

    print("\n=== 4.3 Method disagreement > 7 days (share of active pixel-years) ===")
    for ph, cl in (("FS", "39 before / 39 after"), ("MS", "44 before / 43 after")):
        dd = np.abs(D[(ph, "dynamic")] - D[(ph, "static")])[:, act[ph]]
        pre = np.nanmean(dd[ipre] > 7) * 100
        post = np.nanmean(dd[ipost] > 7) * 100
        line(f"{ph}: disagreement before / after (%)", f"{pre:.0f}/{post:.0f}", cl)

    print("\n=== 4.4 Trends: median over active pixels (days per year) ===")
    for ph, cl in (("FS", "pre -0.38 dyn / -0.35 sta; post +1.60 dyn / +1.38 sta"),
                   ("MS", "pre +0.08; post +0.07 (dynamic)")):
        vals = []
        for m_ in ("dynamic", "static"):
            sp = slope(D[(ph, m_)][ipre], Y[ipre])[act[ph]]
            so = slope(D[(ph, m_)][ipost], Y[ipost])[act[ph]]
            vals.append(f"{m_[:3]} pre {np.nanmedian(sp):+.2f} post {np.nanmedian(so):+.2f}")
        print(f"  {ph}: " + " | ".join(vals) + f"   text: {cl}")
    print("  full-record sign agreement (text: FS 44 earlier / 42 later; MS 63 earlier; before 2016 FS 75 earlier,"
          " after 64 later)")
    for ph in ("FS", "MS"):
        sd = slope(D[(ph, "dynamic")], Y)[act[ph]]
        ss = slope(D[(ph, "static")], Y)[act[ph]]
        pd_ = slope(D[(ph, "dynamic")][ipre], Y[ipre])[act[ph]]
        ps = slope(D[(ph, "static")][ipre], Y[ipre])[act[ph]]
        qd = slope(D[(ph, "dynamic")][ipost], Y[ipost])[act[ph]]
        qs = slope(D[(ph, "static")][ipost], Y[ipost])[act[ph]]
        print(f"    {ph}: full both-earlier {100*np.mean((sd<0)&(ss<0)):.1f}%, both-later {100*np.mean((sd>0)&(ss>0)):.1f}%;"
              f" pre both-earlier {100*np.mean((pd_<0)&(ps<0)):.1f}%; post both-later {100*np.mean((qd>0)&(qs>0)):.1f}%")


if __name__ == "__main__":
    main()
