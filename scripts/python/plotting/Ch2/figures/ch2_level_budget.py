#!/usr/bin/env python3
"""
ch2_level_budget.py -- where did the lost extent go?

Partitions the post-2016 change in annual-mean sea ice extent by zone, by season,
and by cause. Extent is the area of cells with SIC >= 15%, so the annual-mean extent
at a cell is (days above 15% per year) x cell area / 365. The change in annual-mean
extent is then the sum over cells of the change in those days, which can be split by
the zones already used in Ch2 (fringe, mid-season, coastal band, interior) and by month.

Causes (per cell, per year), applied to the days lost:
  summer     days lost in Feb-Mar at cells that survived most summers in the baseline
             (the summer-ice loss of Fig. S3 / the zone map)
  no-freeze  whole-year loss at fringe cells with no 5-day run above 15% that year
  season     everything else: later advance and earlier melt at cells that still freeze

Inputs (defaults are the project's files):
  --nc        data/merged/merged_bootstrap_SH_latest.nc   (N07_ICECON, daily, 25 km)
  --zones     results/ch2_zone_masks.npz                  (short, mid, long, foot)
  --landmask  data/landmask_sic.nc                        (land)
  --survival  results/ch2_summer_survival.nc              (year, y, x)
  --sectors   data/canonical_sectors.nc                   (sector_id)

Output: prints the budget and writes results/ch2_level_budget.csv (one row per zone x month
and the totals). Compare the total with the RAW level step in Ch3 (about -0.91 x 10^6 km^2),
not the detrended -1.1: this budget is raw. It will not close exactly: the level is circumpolar
extent, the zones cover the fixed 1979-2015 footprint, and the periods are means not fits.

Usage (from processing/, seaice env; takes a few minutes, reads one year at a time):
  python ch2_level_budget.py
  python ch2_level_budget.py --pre 1979 2015 --post 2016 2023
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt

ROOT = Path("/user/geog/falejandraperez/sea-ice-phase")
TH, K, CELL = 0.15, 5, 625.0  # 15 %, 5-day run, km^2 per 25 km cell
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def unpack(da):
    v = da.where(da < 1050)
    if float(v.max()) > 1.5:
        v = v / 1000.0
    return v

def has_run(above, k=K):
    """True where the time axis (axis 0) has at least one run of k consecutive True."""
    a = above.astype(np.int8)
    if a.shape[0] < k:
        return np.zeros(a.shape[1:], bool)
    c = np.cumsum(a, axis=0)
    c = np.concatenate([np.zeros((1,) + a.shape[1:], a.dtype), c], axis=0)
    return ((c[k:] - c[:-k]) == k).any(axis=0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", default=str(ROOT / "data/merged/merged_bootstrap_SH_latest.nc"))
    p.add_argument("--zones", default=str(ROOT / "results/ch2_zone_masks.npz"))
    p.add_argument("--landmask", default=str(ROOT / "data/landmask_sic.nc"))
    p.add_argument("--survival", default=str(ROOT / "results/ch2_summer_survival.nc"))
    p.add_argument("--sectors", default=str(ROOT / "data/canonical_sectors.nc"))
    p.add_argument("--coast-km", type=float, default=100.0)
    p.add_argument("--pre", nargs=2, type=int, default=[1979, 2015])
    p.add_argument("--post", nargs=2, type=int, default=[2016, 2023])
    p.add_argument("--out", default=str(ROOT / "results/ch2_level_budget.csv"))
    a = p.parse_args()

    z = np.load(a.zones)
    short, mid, long_ = (z[k].astype(bool) for k in ("short", "mid", "long"))
    land = xr.open_dataset(a.landmask)["land"].values.astype(bool)
    coastal = long_ & (distance_transform_edt(~land) * 25.0 <= a.coast_km)
    interior = long_ & ~coastal
    zones = {"fringe": short, "mid-season": mid, "coastal band": coastal, "interior": interior}
    foot = short | mid | long_
    other = ~foot & ~land  # ice outside the fixed footprint (new ice, or ice in years outside the baseline)
    zones["outside footprint"] = other

    sv = xr.open_dataset(a.survival)["survival"]
    yrs_sv = sv["year"].values
    survived = sv.sel(year=(yrs_sv >= a.pre[0]) & (yrs_sv <= a.pre[1])).mean("year").values >= 0.5

    try:
        sds = xr.open_dataset(a.sectors); sec = sds["sector_id"].values
        secnames = {int(i): str(sds[f"sector_{i}_name"].values) for i in range(1, 6)}
    except Exception as e:
        sec, secnames = None, {}; print("sector file not read:", e)

    ds = xr.open_dataset(a.nc)
    t = pd.to_datetime(ds.time.values)
    years = list(range(a.pre[0], a.post[1] + 1))

    # per year: days above 15 % per cell, by month; and the no-freeze flag at fringe cells
    days = {}      # year -> (12, ny, nx) days above threshold per month
    nofreeze = {}  # year -> (ny, nx) bool
    for y in years:
        sel = np.where(t.year == y)[0]
        if sel.size < 300:
            print(f"{y}: only {sel.size} days, skipped"); continue
        c = unpack(ds["N07_ICECON"].isel(time=sel)).values
        above = np.nan_to_num(c, nan=0.0) >= TH
        # SMMR years are every other day: scale each day's weight so a year sums to ~365
        w = 365.0 / sel.size
        mon = t[sel].month
        d = np.zeros((12,) + above.shape[1:])
        for m in range(1, 13):
            d[m - 1] = above[mon == m].sum(axis=0) * w
        days[y] = d
        nofreeze[y] = ~has_run(above) & short
        print(f"{y}: mean extent {d.sum(axis=0)[~land].sum() * CELL / 365 / 1e6:.2f} x10^6 km^2", flush=True)

    pre_y = [y for y in years if a.pre[0] <= y <= a.pre[1] and y in days]
    post_y = [y for y in years if a.post[0] <= y <= a.post[1] and y in days]
    D_pre = np.mean([days[y] for y in pre_y], axis=0)   # (12, ny, nx) mean days per month
    D_post = np.mean([days[y] for y in post_y], axis=0)
    dD = D_post - D_pre                                   # change in days per month per cell
    to_ext = CELL / 365.0 / 1e6                           # days -> 10^6 km^2 of annual-mean extent

    print("\nAnnual-mean extent, all ocean cells:")
    tot_pre, tot_post = D_pre.sum(0)[~land].sum() * to_ext, D_post.sum(0)[~land].sum() * to_ext
    print(f"  {a.pre[0]}-{a.pre[1]}: {tot_pre:.2f}   {a.post[0]}-{a.post[1]}: {tot_post:.2f}   change {tot_post - tot_pre:+.2f} x10^6 km^2")
    print("  (Ch3 raw level step about -0.91; detrended -1.1)")

    rows = []
    print("\nBy zone (change in annual-mean extent, 10^6 km^2):")
    for name, m in zones.items():
        tot = dD.sum(0)[m].sum() * to_ext
        bym = [dD[i][m].sum() * to_ext for i in range(12)]
        print(f"  {name:18s} {tot:+.3f}   " + " ".join(f"{MONTHS[i]} {bym[i]:+.2f}" for i in range(12)))
        for i in range(12):
            rows.append(dict(zone=name, month=MONTHS[i], change=bym[i]))
        rows.append(dict(zone=name, month="year", change=tot))

    # causes
    summer_cells = survived & ~land
    summer = dD[[0, 1, 2]][:, summer_cells].sum() * to_ext            # Jan-Mar at cells that used to survive
    nf_pre = np.mean([nofreeze[y] for y in pre_y], axis=0); nf_post = np.mean([nofreeze[y] for y in post_y], axis=0)
    # extent lost to no-freeze years: (change in no-freeze frequency) x baseline season at those cells
    nofreeze_loss = -((nf_post - nf_pre) * D_pre.sum(0))[short].sum() * to_ext
    total = dD.sum(0)[~land].sum() * to_ext
    season = total - summer - nofreeze_loss
    print("\nBy cause (10^6 km^2):")
    print(f"  summer ice lost (Jan-Mar, cells that survived >= half of baseline summers): {summer:+.3f}")
    print(f"  fringe years with no freeze-up at all (no 5-day run above 15 %):            {nofreeze_loss:+.3f}")
    print(f"  remainder: later advance / earlier melt where ice still forms:              {season:+.3f}")
    print(f"  total:                                                                       {total:+.3f}")
    print(f"  no-freeze frequency at fringe cells: {nf_pre[short].mean() * 100:.1f} % -> {nf_post[short].mean() * 100:.1f} % of cell-years")
    for k, v in [("summer", summer), ("no-freeze", nofreeze_loss), ("season", season), ("total", total)]:
        rows.append(dict(zone="cause", month=k, change=v))

    if sec is not None:
        print("\nBy sector:")
        for v, lab in secnames.items():
            m = (sec == v) & ~land
            print(f"  {lab:28s} {dD.sum(0)[m].sum() * to_ext:+.3f}")
            rows.append(dict(zone="sector " + lab, month="year", change=dD.sum(0)[m].sum() * to_ext))

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("\nwrote", a.out)

if __name__ == "__main__":
    main()
