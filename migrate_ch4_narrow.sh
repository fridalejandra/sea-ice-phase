#!/usr/bin/env bash
#
# migrate_ch4_narrow.sh
#
# NARROW migration: moves ONLY the unambiguous Ch4 keepers (pipeline scripts,
# analysis scripts, result tables). Leaves ALL figure scripts in scar_poster
# for later sorting. Uses plain `mv` because the scar_poster files are UNTRACKED
# in git (git mv would fail on them). After this runs, `git add` the new tree.
#
# Run from repo root, ON THE ch4-writeup BRANCH:
#   bash migrate_ch4_narrow.sh          # dry run
#   bash migrate_ch4_narrow.sh --go     # actually move
#
set -euo pipefail
DRY=1; [ "${1:-}" = "--go" ] && DRY=0

S="scripts/python/scar_poster"
CH4="scripts/python/plotting/Ch4"
RES="results/ch4"

mv_safe() {  # mv only if source exists; report clearly
  local src="$1" dst="$2"
  if [ ! -e "$src" ]; then echo "  [MISSING] $src -- skipped"; return; fi
  if [ "$DRY" = "1" ]; then echo "  [dry] mv $src -> $dst";
  else mkdir -p "$(dirname "$dst")"; mv "$src" "$dst"; echo "  moved $src"; fi
}

echo "=== dirs ==="
for d in "$CH4/pipeline" "$CH4/analysis" "$RES/tables" "$RES/derived_nc" "$RES/figures"; do
  if [ "$DRY" = "1" ]; then echo "  [dry] mkdir -p $d"; else mkdir -p "$d"; fi
done

echo; echo "=== pipeline ==="
mv_safe "$S/pipeline/compute_ice_divergence_nsidc0116.py" "$CH4/pipeline/01_compute_divergence.py"
mv_safe "$S/pipeline/add_latlon_to_ease_divergence.py"    "$CH4/pipeline/02_add_latlon_divergence.py"
mv_safe "$S/regrid_wind_to_ease.py"                       "$CH4/pipeline/03_regrid_wind_to_ease.py"

echo; echo "=== analysis ==="
mv_safe "$S/analysis/wind_divergence_coupling_test.py"     "$CH4/analysis/01_sector_coupling.py"
mv_safe "$S/analysis/wind_sensitivity_interaction_test.py" "$CH4/analysis/02_sensitivity_interaction.py"
mv_safe "$S/analysis/persistence_efold.py"                 "$CH4/analysis/persistence_efold.py"
mv_safe "$S/analysis/extract_interaction_residuals.py"     "$CH4/analysis/extract_interaction_residuals.py"
mv_safe "$S/analysis/trend_analysis_sector-month-season.py" "$CH4/analysis/trend_analysis.py"

echo; echo "=== result tables ==="
mv_safe "$S/ice_divergence_by_sector_season.csv"          "$RES/tables/divergence_by_sector_season.csv"
mv_safe "$S/wind_divergence_oceanstate_test.csv"          "$RES/tables/sector_coupling_results.csv"
mv_safe "$S/wind_divergence_oceanstate_test_div_positive.csv" "$RES/tables/sector_coupling_results_div_positive.csv"
mv_safe "$S/wind_divergence_oceanstate_test_div_negative.csv" "$RES/tables/sector_coupling_results_div_negative.csv"
mv_safe "$S/wind_divergence_binary_test.csv"             "$RES/tables/sector_coupling_binary.csv"
mv_safe "$S/wind_divergence_binary_test_div_positive.csv" "$RES/tables/sector_coupling_binary_div_positive.csv"
mv_safe "$S/wind_divergence_binary_test_div_negative.csv" "$RES/tables/sector_coupling_binary_div_negative.csv"

echo; echo "=== DONE ($([ $DRY = 1 ] && echo DRY-RUN || echo APPLIED)) ==="
echo "LEFT IN scar_poster (sort later): all fig_*.py, shared pipeline scripts"
echo "  (compute_sia, merge_smmr_patched, regrid_SIC_to_ease, process_era5_sst,"
echo "   fetch_*, mask_*, rebuild_*, add_latlon_to_bootstrap_sic), explore_*,"
echo "   compute_dsic_daily, sst_anomaly_*.csv, explore_*.csv"
if [ "$DRY" = "0" ]; then
  echo; echo "NEXT: git add scripts/python/plotting/Ch4 results/ch4 && git commit -m 'Ch4: migrate keepers from scar_poster'"
fi
