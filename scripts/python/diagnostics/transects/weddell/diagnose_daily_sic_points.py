#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_daily_sic_points.py

Compute daily SIC time series + threshold-free diagnostics for selected grid points.

Inputs:
  --sic         merged daily SIC NetCDF (Bootstrap style; flags like 1100/1200)
  --var         SIC variable name (default N07_ICECON)
  --points-csv  CSV with columns: point,fraction,x_idx,y_idx (from your transect script)
  --outdir      output directory

Optional:
  --start/--end  subset time range (YYYY-MM-DD)
  --lags         max autocorr lag (default 10)
  --nbins        histogram bins for SIC and ΔSIC (default 40)
  --qlo/--qhi    quantile cutoffs for low/high states (default 0.2/0.8)
  --jump-thresh  comma thresholds for |ΔSIC| jump counts (default 0.2,0.4,0.6)
  --make-plots   quick-look plots (timeseries + monthly ΔSIC violin-like via boxplot)
                (kept minimal, because you’ll likely plot later)
Outputs (CSV):
  1) <prefix>_daily_long.csv
  2) <prefix>_daily_wide.csv
  3) <prefix>_monthly_summary.csv
  4) <prefix>_monthly_hist_sic.csv
  5) <prefix>_monthly_hist_dsic.csv
  6) <prefix>_monthly_autocorr.csv
  7) <prefix>_monthly_runlengths.csv
  8) <prefix>_monthly_signstats.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


# --------------------------
# utilities
# --------------------------
def _month_index(dates: pd.DatetimeIndex) -> np.ndarray:
    return dates.month.values


def _safe_autocorr(x: np.ndarray, max_lag: int) -> dict:
    """
    Autocorrelation for lags 1..max_lag ignoring NaNs.
    Returns dict {lag: r} with NaN if insufficient data.
    """
    out = {}
    x = np.asarray(x, float)

    for lag in range(1, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            out[lag] = np.nan
            continue
        aa = a[m] - np.nanmean(a[m])
        bb = b[m] - np.nanmean(b[m])
        denom = np.sqrt(np.nansum(aa * aa) * np.nansum(bb * bb))
        out[lag] = np.nan if denom == 0 else float(np.nansum(aa * bb) / denom)
    return out


def _run_lengths(mask: np.ndarray) -> list:
    """
    Given boolean array mask, return lengths of consecutive True runs.
    NaNs should be pre-handled before calling.
    """
    mask = np.asarray(mask, bool)
    if mask.size == 0:
        return []
    # find run boundaries
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    return (ends - starts).tolist()


def _hist_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.zeros(len(bins) - 1, dtype=int)
    return np.histogram(v, bins=bins)[0]


def _sign_flips(signs: np.ndarray) -> int:
    """
    Count number of sign changes in a sign series (values in {-1, 0, +1}).
    Ignores zeros by collapsing them out.
    """
    s = np.asarray(signs, int)
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


# --------------------------
# main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sic", required=True, type=Path)
    ap.add_argument("--var", default="N07_ICECON")
    ap.add_argument("--points-csv", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--prefix", default="weddell_lon30", help="prefix for output files")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--lags", type=int, default=10)
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--qlo", type=float, default=0.2)
    ap.add_argument("--qhi", type=float, default=0.8)
    ap.add_argument("--jump-thresh", default="0.2,0.4,0.6")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(args.points_csv)
    for c in ["point", "fraction", "x_idx", "y_idx"]:
        if c not in pts.columns:
            raise RuntimeError(f"points CSV missing '{c}'. Found columns: {list(pts.columns)}")
    pts["x_idx"] = pts["x_idx"].astype(int)
    pts["y_idx"] = pts["y_idx"].astype(int)

    jump_thresh = [float(x) for x in args.jump_thresh.split(",") if x.strip()]

    ds = xr.open_dataset(args.sic)
    if args.var not in ds.data_vars:
        raise RuntimeError(f"{args.var} not found in {args.sic}. Available: {list(ds.data_vars)}")

    sic_raw = ds[args.var]
    if not {"time", "y", "x"}.issubset(set(sic_raw.dims)):
        raise RuntimeError(f"SIC dims {sic_raw.dims} do not include ('time','y','x').")

    # time subset
    if args.start or args.end:
        t0 = args.start if args.start else str(pd.to_datetime(sic_raw["time"].values[0]).date())
        t1 = args.end if args.end else str(pd.to_datetime(sic_raw["time"].values[-1]).date())
        sic_raw = sic_raw.sel(time=slice(t0, t1))

    times = pd.to_datetime(sic_raw["time"].values)
    months = times.month.values

    # mask flags and non-physical values
    sic = sic_raw.where(sic_raw < 100)                 # remove 1100/1200/etc
    sic = sic.where((sic >= 0.0) & (sic <= 1.0))       # enforce 0..1

    # Bins
    sic_bins = np.linspace(0.0, 1.0, args.nbins + 1)
    dsic_bins = np.linspace(-1.0, 1.0, args.nbins + 1)

    # --- build daily long + wide
    daily_rows = []
    wide = {"date": times}

    # store per point arrays for monthly stats
    per_point = {}

    for _, r in pts.iterrows():
        name = str(r["point"])
        frac = float(r["fraction"])
        xi = int(r["x_idx"])
        yi = int(r["y_idx"])

        ts = sic.isel(x=xi, y=yi).values.astype(float)
        dsic = np.full_like(ts, np.nan)
        dsic[1:] = ts[1:] - ts[:-1]

        per_point[name] = {
            "fraction": frac,
            "x_idx": xi,
            "y_idx": yi,
            "sic": ts,
            "dsic": dsic,
        }

        daily_rows.append(pd.DataFrame({
            "date": times,
            "month": months,
            "point": name,
            "fraction": frac,
            "x_idx": xi,
            "y_idx": yi,
            "sic": ts,
            "dsic_1d": dsic,
        }))

        wide[f"sic_{name}"] = ts
        wide[f"dsic_{name}"] = dsic

    df_long = pd.concat(daily_rows, ignore_index=True)
    df_wide = pd.DataFrame(wide)

    out_long = outdir / f"{args.prefix}_daily_long.csv"
    out_wide = outdir / f"{args.prefix}_daily_wide.csv"
    df_long.to_csv(out_long, index=False)
    df_wide.to_csv(out_wide, index=False)
    print(f"[info] wrote {out_long}")
    print(f"[info] wrote {out_wide}")

    # --- monthly diagnostics
    monthly_summary = []
    monthly_hist_sic = []
    monthly_hist_dsic = []
    monthly_acf = []
    monthly_runs = []
    monthly_sign = []

    for point_name, d in per_point.items():
        frac = d["fraction"]
        xi = d["x_idx"]
        yi = d["y_idx"]
        ts = d["sic"]
        dsic = d["dsic"]

        # Quantile thresholds (computed on ALL finite SIC for this point)
        finite = ts[np.isfinite(ts)]
        if finite.size >= 30:
            qlo = float(np.nanquantile(finite, args.qlo))
            qhi = float(np.nanquantile(finite, args.qhi))
        else:
            qlo = np.nan
            qhi = np.nan

        for m in range(1, 13):
            mm = (months == m)
            sic_m = ts[mm]
            dsic_m = dsic[mm]

            n = int(mm.sum())
            n_valid = int(np.isfinite(sic_m).sum())
            frac_valid = np.nan if n == 0 else n_valid / n

            # SIC summary
            sic_mean = float(np.nanmean(sic_m)) if n_valid else np.nan
            sic_std = float(np.nanstd(sic_m)) if n_valid else np.nan
            sic_p05 = float(np.nanpercentile(sic_m, 5)) if n_valid else np.nan
            sic_p50 = float(np.nanpercentile(sic_m, 50)) if n_valid else np.nan
            sic_p95 = float(np.nanpercentile(sic_m, 95)) if n_valid else np.nan

            # ΔSIC summary (need >=2 valid points)
            ds_valid = np.isfinite(dsic_m).sum()
            ds_mean = float(np.nanmean(dsic_m)) if ds_valid else np.nan
            ds_std = float(np.nanstd(dsic_m)) if ds_valid else np.nan
            ds_p01 = float(np.nanpercentile(dsic_m, 1)) if ds_valid else np.nan
            ds_p99 = float(np.nanpercentile(dsic_m, 99)) if ds_valid else np.nan
            ds_absmax = float(np.nanmax(np.abs(dsic_m))) if ds_valid else np.nan

            # jump counts at descriptive thresholds
            jump_counts = {}
            for th in jump_thresh:
                jump_counts[f"count_abs_dsic_gt_{th:g}"] = int(np.nansum(np.abs(dsic_m) > th)) if ds_valid else 0

            # histograms
            h_sic = _hist_counts(sic_m, sic_bins)
            h_ds = _hist_counts(dsic_m, dsic_bins)

            for bi in range(len(sic_bins) - 1):
                monthly_hist_sic.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "bin_left": float(sic_bins[bi]),
                    "bin_right": float(sic_bins[bi+1]),
                    "count": int(h_sic[bi]),
                })

            for bi in range(len(dsic_bins) - 1):
                monthly_hist_dsic.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "bin_left": float(dsic_bins[bi]),
                    "bin_right": float(dsic_bins[bi+1]),
                    "count": int(h_ds[bi]),
                })

            # autocorr on SIC by month
            ac = _safe_autocorr(sic_m, args.lags)
            for lag, rlag in ac.items():
                monthly_acf.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "lag": int(lag), "autocorr": rlag
                })

            # run lengths using quantile-defined states (low/mid/high)
            # only if quantiles exist
            if np.isfinite(qlo) and np.isfinite(qhi) and n_valid >= 10:
                low = np.isfinite(sic_m) & (sic_m <= qlo)
                mid = np.isfinite(sic_m) & (sic_m > qlo) & (sic_m < qhi)
                high = np.isfinite(sic_m) & (sic_m >= qhi)

                low_runs = _run_lengths(low)
                mid_runs = _run_lengths(mid)
                high_runs = _run_lengths(high)

                def _summ(runs):
                    if len(runs) == 0:
                        return (0, np.nan, np.nan, np.nan)
                    rr = np.asarray(runs, float)
                    return (int(len(rr)), float(np.nanmean(rr)), float(np.nanmedian(rr)), float(np.nanmax(rr)))

                low_n, low_mean, low_med, low_max = _summ(low_runs)
                mid_n, mid_mean, mid_med, mid_max = _summ(mid_runs)
                high_n, high_mean, high_med, high_max = _summ(high_runs)

                monthly_runs.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "qlo": qlo, "qhi": qhi,
                    "low_n_runs": low_n, "low_mean_len": low_mean, "low_median_len": low_med, "low_max_len": low_max,
                    "mid_n_runs": mid_n, "mid_mean_len": mid_mean, "mid_median_len": mid_med, "mid_max_len": mid_max,
                    "high_n_runs": high_n, "high_mean_len": high_mean, "high_median_len": high_med, "high_max_len": high_max,
                })
            else:
                monthly_runs.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "qlo": qlo, "qhi": qhi,
                    "low_n_runs": 0, "low_mean_len": np.nan, "low_median_len": np.nan, "low_max_len": np.nan,
                    "mid_n_runs": 0, "mid_mean_len": np.nan, "mid_median_len": np.nan, "mid_max_len": np.nan,
                    "high_n_runs": 0, "high_mean_len": np.nan, "high_median_len": np.nan, "high_max_len": np.nan,
                })

            # sign persistence / flips for ΔSIC (ignore zeros)
            if ds_valid >= 10:
                sgn = np.sign(dsic_m)
                sgn[np.isnan(sgn)] = 0
                sgn = sgn.astype(int)

                flips = _sign_flips(sgn)
                # sign runs (after dropping zeros)
                s = sgn[sgn != 0]
                if s.size == 0:
                    run_mean = np.nan
                    run_med = np.nan
                    run_max = np.nan
                else:
                    # run lengths of consecutive equal signs
                    changes = np.where(s[1:] != s[:-1])[0] + 1
                    starts = np.r_[0, changes]
                    ends = np.r_[changes, s.size]
                    runlens = (ends - starts).astype(float)
                    run_mean = float(np.mean(runlens))
                    run_med = float(np.median(runlens))
                    run_max = float(np.max(runlens))

                monthly_sign.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "sign_flips": flips,
                    "sign_flips_per_valid_day": float(flips / max(1, ds_valid)),
                    "sign_run_mean": run_mean,
                    "sign_run_median": run_med,
                    "sign_run_max": run_max,
                })
            else:
                monthly_sign.append({
                    "point": point_name, "fraction": frac, "month": m,
                    "sign_flips": 0,
                    "sign_flips_per_valid_day": np.nan,
                    "sign_run_mean": np.nan,
                    "sign_run_median": np.nan,
                    "sign_run_max": np.nan,
                })

            monthly_summary.append({
                "point": point_name, "fraction": frac, "x_idx": xi, "y_idx": yi,
                "month": m,
                "n_days": n, "n_valid": n_valid, "frac_valid": frac_valid,
                "sic_mean": sic_mean, "sic_std": sic_std, "sic_p05": sic_p05, "sic_p50": sic_p50, "sic_p95": sic_p95,
                "dsic_mean": ds_mean, "dsic_std": ds_std, "dsic_p01": ds_p01, "dsic_p99": ds_p99, "dsic_absmax": ds_absmax,
                "qlo_global": qlo, "qhi_global": qhi,
                **jump_counts,
            })

    df_ms = pd.DataFrame(monthly_summary)
    df_hs = pd.DataFrame(monthly_hist_sic)
    df_hd = pd.DataFrame(monthly_hist_dsic)
    df_ac = pd.DataFrame(monthly_acf)
    df_rl = pd.DataFrame(monthly_runs)
    df_sg = pd.DataFrame(monthly_sign)

    out_ms = outdir / f"{args.prefix}_monthly_summary.csv"
    out_hs = outdir / f"{args.prefix}_monthly_hist_sic.csv"
    out_hd = outdir / f"{args.prefix}_monthly_hist_dsic.csv"
    out_ac = outdir / f"{args.prefix}_monthly_autocorr.csv"
    out_rl = outdir / f"{args.prefix}_monthly_runlengths.csv"
    out_sg = outdir / f"{args.prefix}_monthly_signstats.csv"

    df_ms.to_csv(out_ms, index=False)
    df_hs.to_csv(out_hs, index=False)
    df_hd.to_csv(out_hd, index=False)
    df_ac.to_csv(out_ac, index=False)
    df_rl.to_csv(out_rl, index=False)
    df_sg.to_csv(out_sg, index=False)

    print(f"[info] wrote {out_ms}")
    print(f"[info] wrote {out_hs}")
    print(f"[info] wrote {out_hd}")
    print(f"[info] wrote {out_ac}")
    print(f"[info] wrote {out_rl}")
    print(f"[info] wrote {out_sg}")

    if args.make_plots:
        # Minimal quick-look plots (kept intentionally simple)
        import matplotlib.pyplot as plt

        # Timeseries for each point (SIC + ΔSIC)
        for point_name, d in per_point.items():
            frac = d["fraction"]
            ts = d["sic"]
            dsic = d["dsic"]

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(times, ts)
            ax.set_title(f"{point_name} (f={frac:.2f}) daily SIC")
            ax.set_ylabel("SIC")
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(outdir / f"{args.prefix}_{point_name}_timeseries_SIC.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(times, dsic)
            ax.set_title(f"{point_name} (f={frac:.2f}) daily ΔSIC")
            ax.set_ylabel("ΔSIC (1 day)")
            ax.set_ylim(-1.0, 1.0)
            fig.tight_layout()
            fig.savefig(outdir / f"{args.prefix}_{point_name}_timeseries_dSIC.png", dpi=150)
            plt.close(fig)

        print(f"[info] wrote quick-look plots to {outdir}")


if __name__ == "__main__":
    main()
