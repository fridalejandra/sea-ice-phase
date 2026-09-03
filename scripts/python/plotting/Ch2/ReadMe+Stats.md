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
=== Window sensitivity distribution stats (|delta date|, days) ===
  FS 3 vs 5 days: n=1280495, median=0.0, p90=6.0, p95=11.0, p99=23.0
  FS 7 vs 5 days: n=1258870, median=0.0, p90=0.0, p95=9.0, p99=21.0
  MS 3 vs 5 days: n=1304340, median=0.0, p90=0.0, p95=7.0, p99=20.0
  MS 7 vs 5 days: n=1304106, median=0.0, p90=0.0, p95=0.0, p99=19.0
```

---

## Sector-mean trends — compute_sector_mean_trends07.py

```
=== Sector-mean linear trend slopes (days/year), active80 pixels ===
phase  sector_id sector_label  method  mean_slope_days_per_yr  median_slope_days_per_yr  std_slope  n_active_pixels
   FS          1          A–B Dynamic                  0.1549                    0.0908     0.3582             1543
   FS          1          A–B  Static                  0.2155                    0.2260     0.3477             1543
   FS          2          WED Dynamic                  0.1069                    0.0613     0.2786             4316
   FS          2          WED  Static                  0.0997                    0.0622     0.2715             4316
   FS          3          KHV Dynamic                 -0.0039                    0.0061     0.2109             8447
   FS          3          KHV  Static                 -0.0068                   -0.0086     0.1973             8447
   FS          4           EA Dynamic                 -0.0326                   -0.0689     0.4226             4130
   FS          4           EA  Static                 -0.0367                   -0.0902     0.3385             4130
   FS          5           RA Dynamic                 -0.1662                   -0.1748     0.3045             5750
   FS          5           RA  Static                 -0.1012                   -0.1029     0.2867             5750
   MS          1          A–B Dynamic                 -0.1875                   -0.1678     0.3742             1218
   MS          1          A–B  Static                 -0.2027                   -0.1651     0.3345             1218
   MS          2          WED Dynamic                 -0.2585                   -0.2011     0.2648             3885
   MS          2          WED  Static                 -0.2410                   -0.1643     0.2764             3885
   MS          3          KHV Dynamic                 -0.1417                   -0.1358     0.2071             8461
   MS          3          KHV  Static                 -0.1544                   -0.1371     0.1845             8461
   MS          4           EA Dynamic                 -0.0807                   -0.0848     0.4060             3791
   MS          4           EA  Static                 -0.1492                   -0.1466     0.3910             3791
   MS          5           RA Dynamic                  0.0136                    0.0179     0.2754             5745
   MS          5           RA  Static                 -0.0292                   -0.0124     0.2747             5745

=== Range summary (for manuscript text) ===
FS Dynamic: -0.166 to +0.155 days/yr across sectors
FS Static: -0.101 to +0.215 days/yr across sectors
MS Dynamic: -0.258 to +0.014 days/yr across sectors
MS Static: -0.241 to -0.029 days/yr across sectors

=== Sign agreement check, MS static vs dynamic, per sector ===
method        Dynamic  Static  same_sign
sector_label                            
A–B           -0.1875 -0.2027       True
EA            -0.0807 -0.1492       True
KHV           -0.1417 -0.1544       True
RA             0.0136 -0.0292      False
WED           -0.2585 -0.2410       True
```

Text claims this affects: sector-mean trend slopes (days/yr), MS sign
agreement 4/5 sectors (RA splits, dyn+0.014/sta-0.029), FS agrees 5/5.

---

## Four-diagnostic null-variance result — diagnostic_variance_crossing.py
```
========================================================================
PART 1: Pre/post-2016 variance of phase-date anomalies (Levene's test)
========================================================================
phase  method sector  pre_std  post_std  std_ratio_post_over_pre  levene_p  n_pre  n_post
   FS Dynamic    A–B  26.0181   29.3739                   1.1290    0.0000  50729   13448
   FS Dynamic    WED  22.7511   20.4675                   0.8996    0.0000 144355   37426
   FS Dynamic    KHV  16.1107   17.3682                   1.0781    0.0000 283455   73428
   FS Dynamic     EA  23.0045   23.1863                   1.0079    0.0000 135547   35288
   FS Dynamic     RA  25.0442   22.9813                   0.9176    0.0000 189564   48936
   FS  Static    A–B  25.4129   28.4967                   1.1213    0.0000  50445   13548
   FS  Static    WED  22.8364   19.8644                   0.8699    0.0000 142637   37859
   FS  Static    KHV  15.1207   16.4711                   1.0893    0.0000 282893   73644
   FS  Static     EA  19.9225   20.4326                   1.0256    0.0000 136019   35844
   FS  Static     RA  22.6664   20.8412                   0.9195    0.0000 189808   50189
   MS Dynamic    A–B  35.6619   34.5339                   0.9684    0.0000  40718   10838
   MS Dynamic    WED  24.9298   23.1571                   0.9289    0.0000 128923   34402
   MS Dynamic    KHV  18.9232   20.2377                   1.0695    0.0000 285527   74672
   MS Dynamic     EA  25.8684   23.6149                   0.9129    0.0000 127061   33634
   MS Dynamic     RA  28.1635   28.9240                   1.0270    0.4860 192789   51288
   MS  Static    A–B  35.3438   33.2823                   0.9417    0.0000  40394   10901
   MS  Static    WED  24.1591   21.8902                   0.9061    0.0000 127249   34593
   MS  Static    KHV  18.4354   18.4692                   1.0018    0.8567 285137   74793
   MS  Static     EA  25.2067   24.0070                   0.9524    0.0000 126362   33704
   MS  Static     RA  26.9123   25.9489                   0.9642    0.0000 188469   50970

Note: only 9 post-2016 years feed this test — same thin-sample
caveat that already applies to the Fig. 8 post-2016 trend slopes.

========================================================================
PART 2: Static-vs-dynamic disagreement rate per year (|diff| > 7 days)
========================================================================

FS: pre-2016 mean disagreement rate = 0.399, post-2016 = 0.398, delta = -0.001
1979   0.4130
1980   0.4227
1981   0.4289
1982   0.4756
1983   0.4263
1984   0.4181
1985   0.4011
1986   0.4523
1988   0.4458
1989   0.3653
1990   0.4085
1992   0.4018
1993   0.3930
1994   0.4064
1996   0.3760
1997   0.3763
1998   0.3866
1999   0.3897
2000   0.3963
2001   0.4256
2002   0.4043
2003   0.3798
2004   0.4298
2005   0.3338
2006   0.3651
2007   0.3580
2008   0.4174
2009   0.4082
2010   0.3469
2011   0.3589
2012   0.3805
2013   0.4060
2014   0.3639
2015   0.3981
2016   0.3801
2017   0.3733
2018   0.3693
2019   0.3544
2020   0.3982
2021   0.3827
2022   0.4253
2023   0.4500
2024   0.4469

MS: pre-2016 mean disagreement rate = 0.435, post-2016 = 0.416, delta = -0.019
1979   0.5034
1980   0.4498
1981   0.5589
1982   0.5017
1983   0.5779
1984   0.4953
1985   0.5663
1986   0.6284
1988   0.3807
1989   0.4434
1990   0.4609
1992   0.4246
1993   0.3816
1994   0.4540
1996   0.4252
1997   0.3600
1998   0.4244
1999   0.4117
2000   0.3268
2001   0.4621
2002   0.4007
2003   0.4320
2004   0.3677
2005   0.3279
2006   0.3771
2007   0.4204
2008   0.4218
2009   0.3813
2010   0.4453
2011   0.4292
2012   0.3385
2013   0.3317
2014   0.4968
2015   0.3852
2016   0.3925
2017   0.4342
2018   0.4191
2019   0.3640
2020   0.3965
2021   0.4453
2022   0.4706
2023   0.4300
2024   0.3919

========================================================================
PART 3: Multi-crossing frequency within search window (raw daily SIC)
This part reads the raw daily record and will take longer than Parts 1-2.
========================================================================
/user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch2/processing/diagnostic_variance_crossing.py:198: UserWarning: The specified chunks separate the stored chunks along dimension "time" starting at index 200. This could degrade performance. Instead, consider rechunking after loading.
  ds = xr.open_dataset(RAW_SIC_FILE, chunks={"time": 200})

FS — mean crossing count per active pixel, pre vs. post 2016:
  A–B: pre-2016 = 4.85, post-2016 = 5.08, delta = +0.23
  WED: pre-2016 = 3.07, post-2016 = 2.94, delta = -0.13
  KHV: pre-2016 = 2.38, post-2016 = 2.38, delta = -0.00
  EA: pre-2016 = 4.08, post-2016 = 4.28, delta = +0.21
  RA: pre-2016 = 3.09, post-2016 = 3.16, delta = +0.06
  Saved per-year detail: /user/geog/falejandraperez/sea-ice-phase/results/crossing_frequency_FS.csv

MS — mean crossing count per active pixel, pre vs. post 2016:
  A–B: pre-2016 = 3.30, post-2016 = 3.15, delta = -0.15
  WED: pre-2016 = 2.29, post-2016 = 2.38, delta = +0.09
  KHV: pre-2016 = 1.86, post-2016 = 1.94, delta = +0.08
  EA: pre-2016 = 2.92, post-2016 = 2.96, delta = +0.04
  RA: pre-2016 = 2.13, post-2016 = 2.24, delta = +0.11
  Saved per-year detail: /user/geog/falejandraperez/sea-ice-phase/results/crossing_frequency_MS.csv
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