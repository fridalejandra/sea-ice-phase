import xarray as xr
import numpy as np
import pandas as pd

# Load data
fs_dur_smmr = xr.open_dataarray('data/transition_metrics/SMMR/transition_dur_FS_thr15.nc', decode_times=False)
fs_dur_amsre = xr.open_dataarray('data/transition_metrics/AMSRE/transition_dur_FS_thr15.nc', decode_times=False)
sectors = xr.open_dataset('data/canonical_sectors.nc')

sector_ids = [1, 2, 3, 4, 5]
sector_labels = {1: "A-B", 2: "WED", 3: "KHV", 4: "EA", 5: "RA"}

print("="*80)
print("SENSOR DISAGREEMENT INVESTIGATION")
print("="*80)

# Test 1: SMMR full period (1979-2024)
print("\n1. SMMR FULL PERIOD (1979-2024)")
print("-"*80)
print(f"{'Sector':<8} {'Pre-2016':<12} {'Post-2016':<12} {'Change':<12} {'% Change':<12}")
print("-"*80)

smmr_full_results = {}
for sec_id in sector_ids:
    sec_mask = sectors['sector_id'] == sec_id
    
    pre = fs_dur_smmr.where(fs_dur_smmr.year < 2016).where(sec_mask).mean().values
    post = fs_dur_smmr.where(fs_dur_smmr.year >= 2016).where(sec_mask).mean().values
    change = post - pre
    pct_change = (change / pre * 100) if pre > 0 else 0
    
    smmr_full_results[sector_labels[sec_id]] = {'pre': pre, 'post': post, 'change': change}
    print(f"{sector_labels[sec_id]:<8} {pre:>10.2f}    {post:>10.2f}    {change:>10.2f}    {pct_change:>10.1f}%")

# Test 2: SMMR overlap period only (2003-2024)
print("\n2. SMMR OVERLAP PERIOD ONLY (2003-2024, same as AMSRE)")
print("-"*80)
print(f"{'Sector':<8} {'Pre-2016':<12} {'Post-2016':<12} {'Change':<12} {'% Change':<12}")
print("-"*80)

fs_dur_smmr_overlap = fs_dur_smmr.sel(year=slice(2003, 2024))
smmr_overlap_results = {}
for sec_id in sector_ids:
    sec_mask = sectors['sector_id'] == sec_id
    
    pre = fs_dur_smmr_overlap.where(fs_dur_smmr_overlap.year < 2016).where(sec_mask).mean().values
    post = fs_dur_smmr_overlap.where(fs_dur_smmr_overlap.year >= 2016).where(sec_mask).mean().values
    change = post - pre
    pct_change = (change / pre * 100) if pre > 0 else 0
    
    smmr_overlap_results[sector_labels[sec_id]] = {'pre': pre, 'post': post, 'change': change}
    print(f"{sector_labels[sec_id]:<8} {pre:>10.2f}    {post:>10.2f}    {change:>10.2f}    {pct_change:>10.1f}%")

# Test 3: AMSRE only (2003-2024, but note it's 2012-2024)
print("\n3. AMSRE PERIOD (2012-2024, shorter window)")
print("-"*80)
print(f"{'Sector':<8} {'Pre-2016':<12} {'Post-2016':<12} {'Change':<12} {'% Change':<12}")
print("-"*80)

amsre_results = {}
for sec_id in sector_ids:
    sec_mask = sectors['sector_id'] == sec_id
    
    # AMSRE is 2012-2024, so "pre-2016" is only 2012-2015 (4 years)
    pre = fs_dur_amsre.where(fs_dur_amsre.year < 2016).where(sec_mask).mean().values
    post = fs_dur_amsre.where(fs_dur_amsre.year >= 2016).where(sec_mask).mean().values
    change = post - pre
    pct_change = (change / pre * 100) if pre > 0 else 0
    
    amsre_results[sector_labels[sec_id]] = {'pre': pre, 'post': post, 'change': change}
    print(f"{sector_labels[sec_id]:<8} {pre:>10.2f}    {post:>10.2f}    {change:>10.2f}    {pct_change:>10.1f}%")

# Test 4: Compare SMMR overlap to AMSRE
print("\n4. DIRECT COMPARISON: SMMR (2003-2024) vs AMSRE (2012-2024)")
print("-"*80)
print(f"{'Sector':<8} {'SMMR Δ':<12} {'AMSRE Δ':<12} {'Agreement?':<20}")
print("-"*80)

for sec in sector_labels.values():
    smmr_delta = smmr_overlap_results[sec]['change']
    amsre_delta = amsre_results[sec]['change']
    
    # Are they in the same direction?
    agree = "YES (same direction)" if smmr_delta * amsre_delta > 0 else "NO (opposite)"
    
    print(f"{sec:<8} {smmr_delta:>10.2f}    {amsre_delta:>10.2f}    {agree:<20}")

# Test 5: Does sharpening come from early data (pre-2003)?
print("\n5. SOURCE OF SMMR SHARPENING")
print("-"*80)
print("Where does the SMMR sharpening signal come from?")

pre_2003 = fs_dur_smmr.sel(year=slice(1979, 2002))
pre_2003_mean = pre_2003.mean().values

post_2016_mean = fs_dur_smmr.where(fs_dur_smmr.year >= 2016).mean().values

full_change = post_2016_mean - fs_dur_smmr.where(fs_dur_smmr.year < 2016).mean().values
early_change = pre_2003_mean - fs_dur_smmr.where(fs_dur_smmr.year >= 2003).mean().values

print(f"Circumpolar mean 1979-2002:     {pre_2003_mean:.2f} days")
print(f"Circumpolar mean 2003-2015:     {fs_dur_smmr.where((fs_dur_smmr.year >= 2003) & (fs_dur_smmr.year < 2016)).mean().values:.2f} days")
print(f"Circumpolar mean 2016-2024:     {post_2016_mean:.2f} days")
print(f"\nFull period change (1979-2024): {full_change:.2f} days")
print(f"Early data contribution (1979-2002 vs 2003+): {early_change:.2f} days")
print(f"Post-2016 contribution: {post_2016_mean - fs_dur_smmr.where((fs_dur_smmr.year >= 2003) & (fs_dur_smmr.year < 2016)).mean().values:.2f} days")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
If SMMR sharpening is STRONGER in overlap period (2003-2024):
→ Real post-2016 regime shift (not just long-term trend)
→ AMSRE disagreement is puzzling (sensor artifact? too short window?)

If SMMR sharpening is WEAKER in overlap period (2003-2024):
→ Most sharpening comes from 1979-2002 (long-term trend)
→ Post-2016 "step" may be smaller than full-period average suggests
→ AMSRE agreement makes sense (both show minimal post-2016 change)

If SMMR 2003-2024 sharpens but AMSRE doesn't:
→ Sensor-specific response to post-2016 conditions
→ Bootstrap vs AMSR2 retrieve SIC differently in recent regime
→ Need to investigate raw SIC, not just duration
""")

