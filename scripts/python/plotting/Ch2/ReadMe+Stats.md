# Ch2 Figure Stats Verification Log


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
Active mask @ 0.80: FS=24186, MS=23100

  --- FS sector median offsets (dynamic - static), active80 ---
    Amundsen– Bellingshausen: median offset = +10.5 days, n=1543
    Weddell: median offset = +6.8 days, n=4316
    King Haakon VII: median offset = +4.8 days, n=8447
    East Antarctica: median offset = +11.5 days, n=4130
    Ross– Amundsen: median offset = +10.1 days, n=5750
  static-valid/dynamic-invalid pixels (whole domain): 1473
    Amundsen– Bellingshausen: 41
    Weddell: 1274
    King Haakon VII: 9
    East Antarctica: 119
    Ross– Amundsen: 30

  --- MS sector median offsets (dynamic - static), active80 ---
    Amundsen– Bellingshausen: median offset = -4.5 days, n=1218
    Weddell: median offset = -10.5 days, n=3885
    King Haakon VII: median offset = -7.2 days, n=8461
    East Antarctica: median offset = -7.6 days, n=3791
    Ross– Amundsen: median offset = -12.4 days, n=5745
  static-valid/dynamic-invalid pixels (whole domain): 1682
    Amundsen– Bellingshausen: 106
    Weddell: 434
    King Haakon VII: 302
    East Antarctica: 499
    Ross– Amundsen: 341
```

---

## Fig 7 — fig07_trends_static_dynamic.py


```
MS_dynamic_clim min/max: 4.0/192.0
MS_static_clim  min/max: 1.0/193.0
Active mask @ 0.80: FS=24186, MS=23100

Sector-mean step-change deltas (post-pre, active-only):
phase  sector_id sector_label  method      delta
   FS          1          A–B Dynamic   3.834116
   FS          1          A–B  Static   4.922143
   FS          2          WED Dynamic   8.539152
   FS          2          WED  Static   8.832134
   FS          3          KHV Dynamic   9.159275
   FS          3          KHV  Static   8.810742
   FS          4           EA Dynamic   5.614586
   FS          4           EA  Static   4.871596
   FS          5           RA Dynamic   7.216149
   FS          5           RA  Static   7.234293
   MS          1          A–B Dynamic  -0.518216
   MS          1          A–B  Static  -0.634838
   MS          2          WED Dynamic -14.017342
   MS          2          WED  Static -12.071669
   MS          3          KHV Dynamic  -8.124432
   MS          3          KHV  Static  -7.870877
   MS          4           EA Dynamic  -6.650866
   MS          4           EA  Static  -7.389708
   MS          5           RA Dynamic  -5.142918
   MS          5           RA  Static  -4.784264
  FS delta range across sectors/methods: +3.8 to +9.2 days
  MS delta range across sectors/methods: -14.0 to -0.5 days
Trend years span: 1979–2024
Step-change BOTH EARLIER fraction (FS) [denom=ACTIVE @ 0.80]: 0.01310675597453072
Step-change BOTH EARLIER fraction (MS) [denom=ACTIVE @ 0.80]: 0.4325108225108225
Trend BOTH NEG SLOPE fraction (FS) [denom=ACTIVE @ 0.80]: 0.45083932853717024
Trend BOTH NEG SLOPE fraction (MS) [denom=ACTIVE @ 0.80]: 0.626969696969697

Pre/post-2016-ONLY trend sign agreement (new — verify against draft text):
  FS pre-2016: both-earlier(neg)=76.5%, both-later(pos)=14.9%  (n=24186, years 1979-2015)
  FS post-2016: both-earlier(neg)=21.7%, both-later(pos)=64.1%  (n=24186, years 2016-2024)
  MS pre-2016: both-earlier(neg)=29.1%, both-later(pos)=55.3%  (n=23100, years 1979-2015)
  MS post-2016: both-earlier(neg)=42.1%, both-later(pos)=41.7%  (n=23100, years 2016-2024)
Trend BOTH POS SLOPE fraction (FS) [denom=ACTIVE @ 0.80]: 0.4059786653435872
Trend BOTH POS SLOPE fraction (MS) [denom=ACTIVE @ 0.80]: 0.20432900432900433
Step-change BOTH LATER fraction (FS) [denom=ACTIVE @ 0.80]: 0.46572397254610104
Step-change BOTH LATER fraction (MS) [denom=ACTIVE @ 0.80]: 0.03666666666666667
```
---

## Fig 8 — fig08_prepost_trends.py
Patch: none needed, already prints median slope per phase/period.
Status: [ ] not run  [ ] run, pasted below  [ ] text updated to match

```
Computing slopes...
  FS pre: median=-0.373
  FS post: median=1.600
  MS pre: median=0.091
  MS post: median=0.067
  DUR pre: median=0.454
  DUR post: median=-1.600
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
- [ ] 46-year vs 43-year record language -- decide convention, apply
      consistently (Fig 4/5/8 paragraphs all currently mix these)
- [ ] C15/"volatility" wording + whether a crossing-frequency figure gets
      added (tied to diagnostic_variance_crossing.py Part 3 output)
- [ ] Citation placeholders still open: MYI (Fig 4), ABS wind-driven
      (Fig 5), Ross Sea wind-driven (Fig 5), static Melt Start reversal
      "as ice thins across the record (cite)" (Fig 8 paragraph, p.33)