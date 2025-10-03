# Chapter 2 Published Plan (Fri → Tue)

## Friday (today) — Prep & review
- [ ] Review completed sensitivity runs (threshold, persistence).  
- [ ] Identify what needs rerun (coverage masks, leap-year handling).  
- [ ] Organize code/scripts into `scripts/phase/sensitivity/` and `scripts/phase/production/`.  
- [ ] Create output folders: `results/sensitivity/` and `results/production/`.  

---

## Saturday — Sensitivity analysis
- [ ] Re-run sensitivity cases (thresholds 10%, 15%, 30%; persistence 3, 5, 7 days).  
- [ ] Generate Δ-maps (climatology differences vs 15%–5d baseline).  
- [ ] Produce sectoral summary table (% pixels |Δ| > 5 d).  
- [ ] Draft 1–2 sensitivity figures (Δ-maps + histograms).  
- [ ] Write short methods note about sensitivity results.  

---

## Sunday — Fixed-method production
- [ ] Rerun FS/MS/ME with baseline (15%–5d; ME=15%–14d).  
- [ ] Generate climatology maps (FS, MS, ME).  
- [ ] Generate trend maps (FS, MS, ME).  
- [ ] Compute frozen-season duration (FS→MS) climatology + trend.  
- [ ] Export sectoral median time series.  
- [ ] QA: spot-check time series, ensure consistent masks.  

---

## Monday — Documentation & polish
- [ ] Write Methods text for sensitivity analysis and fixed method.  
- [ ] Write captions for sensitivity and production figures.  
- [ ] Assemble “Sensitivity” figure (two Δ-maps) and “Production” figure (climatology maps).  

---

## Tuesday — Dynamic method pilot
- [ ] Compute climatology-relative thresholds (M and E) from 1980–2015.  
- [ ] Apply dynamic method to one test sector, 5 years.  
- [ ] Generate Δ vs static maps.  
- [ ] Draft 2–3 sentences for Discussion on definition sensitivity.  

---

## End-of-Tuesday Deliverables
- [ ] Sensitivity section complete (figures + text).  
- [ ] Final FS/MS/ME climatology + trend figures.  
- [ ] Frozen-season duration figure.  
- [ ] Dynamic method tested + summary text.  
- [ ] Repo clean: `scripts/`, `results/`, `figs/`, `docs/` with updated checklist.  
