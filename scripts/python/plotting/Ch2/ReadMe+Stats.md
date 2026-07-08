# Ch2 Figure Stats Verification Log

One section per figure/script. Paste raw stdout under each as you run it.
Goal: every number in the manuscript text traces to a pasted printout here.

---

## Fig 1 — fig01_sensor_bias_amsre_smmr.py
Patch: print_distribution_stats() for FS/MS bias (added [DATE])
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
  MS (Retreat): n=302638, median=+2.8, IQR=[-25.2, +29.2], mode=+6.0, skew=+0.70 (right-skewed), p90(|bias|)=88.5
  FS (Advance): n=268794, median=+8.0, IQR=[-26.2, +42.0], mode=+8.0, skew=-0.06 (roughly symmetric), p90(|bias|)=85.5
```

Text claims this affects: p.17 caption ("modes near zero"), p.18-19
(MS spike/median/IQR, FS uniform/median/IQR, 90th percentile sentence).

---

## Fig 4/5 — fig04-5_climatology_static_dynamic_maps.py
Patch: none needed, quick_stats() already prints what's used.
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
 [display floor >= 10 yrs] FS: static=34829 (of 104912 valid-ocean), dynamic=33592 (of 104912 valid-ocean), joint (what panel c will show)=33147
FS static: min=52.7, max=258.4, p5=69.3, p95=237.9
FS dynamic: min=70.3, max=258.0, p5=100.9, p95=238.2
 [display floor >= 10 yrs] MS: static=36064 (of 104912 valid-ocean), dynamic=35273 (of 104912 valid-ocean), joint (what panel c will show)=33525
MS static: min=12.8, max=186.5, p5=35.7, p95=161.6
MS dynamic: min=16.4, max=177.9, p5=48.1, p95=146.3
```

Text claims this affects: active80 pixel counts (24,186 FS / 23,100 MS),
min-N=10 pixel counts, FS/MS static/dynamic min/max/p5/p95.

---

## Fig 6 — fig06_climatology_sector_violins.py

```
<paste stdout here>
```

Text claims this affects: sector median offsets (Weddell +16d, Ross +14d,
A-B +9d, KHV +4d for FS; Weddell +16d, A-B +7d, EA +7d, KHV -1d, Ross -4d
for MS), and the "over 1,300 multi-year ice pixels" claim.

---

## Fig 7 — fig07_trends_static_dynamic.py
Status: SCRIPT NOT YET PROVIDED -- highest priority, verifies the most
currently-unverified numbers in the draft.

```
<paste stdout here once script is provided and run>
```

Text claims this affects: step-change sign agreement (FS 47%/1%,
MS 23%/5%), sector-mean step changes (5-11 days FS), full-record trend
sign split (FS 46%/41%), pre-2016 (75%/16%), post-2016 (63%/22%).

---

## Fig 8 — fig08_prepost_trends.py
Patch: none needed, already prints median slope per phase/period.
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
<paste stdout here>
```

Text claims this affects: circumpolar-median FS trend reversal
(-0.37 to +1.60 days/yr), duration mirror (+0.45 to -1.60 days/yr).

---

## Fig S1 — figS01_window_sensitivity_ecdfs.py
Patch: optional (add median/p90 print if exact numbers wanted).
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
<paste stdout here>
```

Text claims this affects: Window section "±2-3 days... up to ~8-10 days"
(currently qualitative, could be tightened to exact percentiles).

---

## Sector-mean trends — compute_sector_mean_trends07.py
Patch: none needed, already prints range summary + sign agreement check.
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
<paste stdout here>
```

Text claims this affects: sector-mean trend slopes (days/yr), MS sign
agreement 4/5 sectors (RA splits, dyn+0.014/sta-0.029), FS agrees 5/5.

---

## Four-diagnostic null-variance result — diagnostic_variance_crossing.py
Patch: none needed, Parts 1-3 already print everything.
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
<paste stdout here>
```

Text claims this affects: variance ratio (0.87-1.13, no consistent sign),
method disagreement (flat-to-down), crossing frequency deltas
(-0.15 to +0.23, mixed sign), the 7-day disagreement threshold.

---

## Open questions carried over from this session
- [ ] Fig 7 script still needed
- [ ] 46-year vs 43-year record language -- decide convention, apply
      consistently (Fig 4/5/8 paragraphs all currently mix these)
- [ ] C15/"volatility" wording + whether a crossing-frequency figure gets
      added (tied to diagnostic_variance_crossing.py Part 3 output)
- [ ] Citation placeholders still open: MYI (Fig 4), ABS wind-driven
      (Fig 5), Ross Sea wind-driven (Fig 5), static Melt Start reversal
      "as ice thins across the record (cite)" (Fig 8 paragraph, p.33)