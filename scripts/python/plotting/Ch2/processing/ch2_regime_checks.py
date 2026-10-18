#!/usr/bin/env python
"""
ch2_regime_checks.py

Cheap regime-shift checks using only the existing v3 climatology + anomaly
files (no new detection). Turns three self-imposed limitations into tests:

  (a) CENSORING   non-detection rate per sector, method and period
  (b) SHOULDERS   dynamic - static offset, pre vs post, inner/middle/outer pack
  (c) TRUNCATION  share of detections within EDGE_DAYS of the search-window bounds
  (d) MASK BIAS   post - pre shift in sector-mean dates under active80 vs a
                  footprint fixed from the baseline period only

Pixel footprint for (a)-(c): valid in >= BASE_FRAC of BASE-period years under
both methods. Defined from the baseline only, so post-2016 losses cannot
remove pixels (no survivor bias).

Units: FS in day of year; MS in days since Aug 15 (_dsa variables).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
ANOM = ROOT / "data" / "anomalies" / "SMMR"
SECT = ROOT / "data" / "canonical_sectors.nc"
OUT = ROOT / "results"

SECTORS = {1: "ABS", 2: "WS", 3: "KHV", 4: "EA", 5: "RAS"}
TAGS = {("FS", "static"): "FS_static_thr15_k5", ("FS", "dynamic"): "FS_dynamic_k5_q70",
        ("MS", "static"): "MS_static_thr15_k5", ("MS", "dynamic"): "MS_dynamic_k5_q70"}
PERIODS = {"pre79": (1979, 2015), "pre88": (1988, 2015), "post": (2016, 2024)}
BASE, BASE_FRAC = "pre88", 0.5
WIN = {"FS": (46, 273), "MS": (0, 197)}   # Feb 15-Sep 30 (DOY); Aug 15-Feb 28 (dsa)
EDGE_DAYS = 7


def _var(path, names):
    with xr.open_dataset(path, decode_times=False) as ds:
        for n in names:
            if n in ds:
                return ds[n].load()
    raise KeyError(f"{names} not in {path}")


def load_dates(phase, method):
    t = TAGS[(phase, method)]
    cs = [f"{t}_clim_dsa", f"{t}_clim"] if phase == "MS" else [f"{t}_clim"]
    an = [f"{t}_anom_dsa", f"{t}_anom"] if phase == "MS" else [f"{t}_anom"]
    clim = _var(ANOM / f"{t}_climatology.nc", cs).values
    anom = _var(ANOM / f"{t}_anomalies.nc", an)
    return clim, clim[None] + anom.values, anom["year"].values


def main():
    with xr.open_dataset(SECT) as m:
        ocean = m["valid_ocean"].astype(bool).values
        sid = m["sector_id"].values

    rows = {k: [] for k in "abcd"}
    for phase in ["FS", "MS"]:
        clim, date, years = {}, {}, None
        for meth in ["static", "dynamic"]:
            clim[meth], date[meth], years = load_dates(phase, meth)
        inp = {p: (years >= a) & (years <= b) for p, (a, b) in PERIODS.items()}

        valid = {m_: np.isfinite(date[m_]) for m_ in date}
        foot = ocean & np.all([valid[m_][inp[BASE]].mean(0) >= BASE_FRAC for m_ in valid], axis=0)
        act80 = ocean & np.all([valid[m_].mean(0) >= 0.8 for m_ in valid], axis=0)

        diff = date["dynamic"] - date["static"]
        pm = {p: np.nanmean(diff[inp[p]], axis=0) for p in PERIODS}
        dm = {(m_, p): np.nanmean(date[m_][inp[p]], axis=0) for m_ in date for p in PERIODS}

        regions = [(SECTORS[s], foot & (sid == s)) for s in SECTORS] + [("ALL", foot)]
        for lab, sel in regions:
            for m_ in ["static", "dynamic"]:
                r = dict(phase=phase, sector=lab, method=m_, n_pix=int(sel.sum()))
                for p in PERIODS:
                    r[f"nodetect_{p}"] = round(1 - valid[m_][inp[p]][:, sel].mean(), 3)
                rows["a"].append(r)

            s = clim["static"][sel]
            lo, hi = np.nanpercentile(s, [100 / 3, 200 / 3])
            early, late = np.zeros_like(sel), np.zeros_like(sel)
            early[sel], late[sel] = s <= lo, s >= hi
            inner, outer = (early, late) if phase == "FS" else (late, early)
            for zone, z in [("all", sel), ("inner", inner), ("middle", sel & ~inner & ~outer), ("outer", outer)]:
                both = z & np.isfinite(pm["pre88"]) & np.isfinite(pm["post"])
                rows["b"].append(dict(
                    phase=phase, sector=lab, zone=zone, n_pix=int(both.sum()),
                    pre79=round(np.nanmean(pm["pre79"][z]), 2),
                    pre88=round(np.nanmean(pm["pre88"][both]), 2),
                    post=round(np.nanmean(pm["post"][both]), 2),
                    change_post_minus_pre88=round(np.nanmean((pm["post"] - pm["pre88"])[both]), 2)))

            w0, w1 = WIN[phase]
            for m_ in ["static", "dynamic"]:
                r = dict(phase=phase, sector=lab, method=m_)
                for p in PERIODS:
                    v = date[m_][inp[p]][:, sel]
                    v = v[np.isfinite(v)]
                    r[f"near_start_{p}"] = round(np.mean(v <= w0 + EDGE_DAYS), 3) if v.size else np.nan
                    r[f"near_end_{p}"] = round(np.mean(v >= w1 - EDGE_DAYS), 3) if v.size else np.nan
                rows["c"].append(r)

        for s in list(SECTORS) + [0]:
            lab = SECTORS.get(s, "ALL")
            for m_ in ["static", "dynamic"]:
                r = dict(phase=phase, sector=lab, method=m_)
                for mname, mk in [("active80", act80), ("footprint", foot)]:
                    sel = mk & ((sid == s) if s else True)
                    both = sel & np.isfinite(dm[(m_, "pre88")]) & np.isfinite(dm[(m_, "post")])
                    r[f"shift_{mname}"] = round(np.nanmean((dm[(m_, "post")] - dm[(m_, "pre88")])[both]), 2)
                    r[f"npix_{mname}"] = int(both.sum())
                rows["d"].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    titles = {
        "a": ("(a) non-detection rate (share of pixel-years with no date), baseline footprint", "ch2_check_nodetection"),
        "b": ("(b) shoulder = dynamic - static (days), pixel means, pre vs post", "ch2_check_shoulders"),
        "c": (f"(c) share of detections within {EDGE_DAYS} d of window start/end", "ch2_check_truncation"),
        "d": ("(d) post - pre88 shift in sector-mean date (days): active80 vs baseline footprint", "ch2_check_mask_bias"),
    }
    for k, (title, fname) in titles.items():
        df = pd.DataFrame(rows[k])
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
        df.to_csv(OUT / f"{fname}.csv", index=False)
    print(f"\nCSVs written to {OUT}")


if __name__ == "__main__":
    main()
