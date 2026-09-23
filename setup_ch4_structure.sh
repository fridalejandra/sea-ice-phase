#!/usr/bin/env bash
#
# setup_ch4_structure.sh
#
# Creates the Ch4 directory skeleton (mirroring Ch2/Ch3) and migrates the
# keeper scripts from scar_poster/ into scripts/python/plotting/Ch4/ with
# interpretable numbered names, using `git mv` so history is preserved.
#
# SAFE BY DEFAULT: runs in --dry-run mode (prints what it WOULD do).
# Review the output, then run with --go to actually move files.
#
#   bash setup_ch4_structure.sh            # dry run, shows the plan
#   bash setup_ch4_structure.sh --go       # actually do it
#
# Run from repo root: /user/geog/falejandraperez/sea-ice-phase
set -euo pipefail

DRY=1
[ "${1:-}" = "--go" ] && DRY=0

SCAR="scripts/python/scar_poster"
CH4="scripts/python/plotting/Ch4"
RES="results/ch4"

run() {
  if [ "$DRY" = "1" ]; then echo "  [dry] $*"; else echo "  $*"; eval "$@"; fi
}

echo "=== 1. Create Ch4 script + results directories ==="
for d in \
  "$CH4/pipeline" "$CH4/analysis" "$CH4/figures" \
  "$RES/figures" "$RES/tables" "$RES/derived_nc"; do
  run "mkdir -p $d"
done

echo
echo "=== 2. Migrate PIPELINE scripts (git mv, keeps history) ==="
run "git mv $SCAR/pipeline/compute_ice_divergence_nsidc0116.py $CH4/pipeline/01_compute_divergence.py"
run "git mv $SCAR/pipeline/add_latlon_to_ease_divergence.py    $CH4/pipeline/02_add_latlon_divergence.py"
run "git mv $SCAR/regrid_wind_to_ease.py                       $CH4/pipeline/03_regrid_wind_to_ease.py"

echo
echo "=== 3. Migrate ANALYSIS scripts ==="
run "git mv $SCAR/analysis/wind_divergence_coupling_test.py     $CH4/analysis/01_sector_coupling.py"
run "git mv $SCAR/analysis/persistence_efold.py                 $CH4/analysis/persistence_efold.py"
run "git mv $SCAR/analysis/extract_interaction_residuals.py     $CH4/analysis/extract_interaction_residuals.py"
run "git mv $SCAR/analysis/trend_analysis_sector-month-season.py $CH4/analysis/trend_analysis.py"
# the new block-test (if you've added it to the repo already, else copy it in)
# run "git mv $SCAR/spatial_coupling_blocktest.py               $CH4/analysis/02_spatial_coupling_blocktest.py"

echo
echo "=== 4. Migrate FIGURE scripts (canonical only) ==="
run "git mv $SCAR/fig_wind_strengthening_v2.py                 $CH4/figures/fig_01_wind_trends.py"
run "git mv $SCAR/fig_results_grid.py                          $CH4/figures/fig_02_sector_beta_grid.py"
run "git mv $SCAR/figures/fig_variance_decomposition.py        $CH4/figures/fig_03_variance_decomp.py"
run "git mv $SCAR/plot_monthly_maps.py                         $CH4/figures/fig_04_divergence_maps.py"
run "git mv $SCAR/fig_methods_schematic.py                     $CH4/figures/fig_methods_schematic.py"
run "git mv $SCAR/figures/figs_datasection.py                  $CH4/figures/fig_00_datasection.py"

echo
echo "=== 5. Migrate result TABLES (loose CSVs -> results/ch4/tables) ==="
run "git mv $SCAR/ice_divergence_by_sector_season.csv          $RES/tables/divergence_by_sector_season.csv"
run "git mv $SCAR/wind_divergence_oceanstate_test.csv          $RES/tables/sector_coupling_results.csv"
run "git mv $SCAR/wind_divergence_binary_test.csv              $RES/tables/sector_coupling_binary.csv"
# variants:
for v in div_positive div_negative; do
  run "git mv $SCAR/wind_divergence_oceanstate_test_${v}.csv    $RES/tables/sector_coupling_results_${v}.csv"
  run "git mv $SCAR/wind_divergence_binary_test_${v}.csv        $RES/tables/sector_coupling_binary_${v}.csv"
done

echo
echo "=== 6. ARCHIVE duplicates & exploratory (move aside, don't delete) ==="
run "mkdir -p $SCAR/archive/ch4_superseded"
for f in \
  fig_wind_strengthening.py fig_wind_strenghtening.py \
  fig_wind_djf_mam.py fig_wind_sia_khv_stacked.py \
  explore_high_wind_sensitivity.py explore_miz_mixture.py; do
  run "git mv $SCAR/$f $SCAR/archive/ch4_superseded/$f 2>/dev/null || true"
done

echo
echo "=== DONE ($([ $DRY = 1 ] && echo DRY-RUN || echo APPLIED)) ==="
if [ "$DRY" = "1" ]; then
  echo "Review the [dry] lines above. If they look right, rerun with --go:"
  echo "    bash setup_ch4_structure.sh --go"
else
  echo "Migrated. NEXT:"
  echo "  1. Fix output paths in the 3 pipeline scripts to write to results/ch4/derived_nc/"
  echo "  2. In 01_compute_divergence.py set WRITE_GRIDDED=True"
  echo "  3. git commit -m 'Ch4: migrate scar_poster -> plotting/Ch4, numbered structure'"
fi
