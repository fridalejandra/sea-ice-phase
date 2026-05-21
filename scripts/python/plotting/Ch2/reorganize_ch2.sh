#!/usr/bin/env bash
# =============================================================================
# reorganize_ch2.sh
#
# Reorganizes Chapter 2 scripts to match the Chapter 3 convention:
#
#   Ch2/
#   ├── figures/
#   │   ├── fig01_*.py  …  fig07_*.py      (in-paper figures)
#   │   ├── figS01_*.py …                  (supplement figures)
#   │   └── scratch/                       (exploratory, not in paper)
#   ├── processing/
#   │   └── compute_*.py  pipeline_*.py  download_*.py  merge_*.py
#   └── utils/
#       └── plot_utils.py  config.py
#
# Run from the ROOT of your sea-ice-phase repo, e.g.:
#   bash scripts/python/plotting/Ch2/reorganize_ch2.sh
#
# Pass --dry-run to preview without moving anything.
# Pass --force  to skip the confirmation prompt.
#
# The script NEVER deletes files — it only moves them.
# =============================================================================

set -euo pipefail

# ── locate this script's directory (works from any cwd) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH2="$SCRIPT_DIR"          # the Ch2/ folder itself

DRY_RUN=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1   ;;
  esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
move() {
  local src="$1" dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "  [SKIP – not found] $src"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] mv $src  →  $dst"
  else
    mv "$src" "$dst"
    echo "  mv $(basename "$src")  →  ${dst#$CH2/}"
  fi
}

divider() { echo ""; echo "── $1 ──"; }

# ── confirmation ──────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 && $FORCE -eq 0 ]]; then
  echo "This will reorganize scripts under:"
  echo "  $CH2"
  echo ""
  read -r -p "Continue? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

[[ $DRY_RUN -eq 1 ]] && echo "[DRY RUN — no files will be moved]"

# =============================================================================
# 1. CREATE TARGET DIRECTORIES
# =============================================================================
for d in figures figures/scratch processing utils; do
  [[ $DRY_RUN -eq 1 ]] && echo "  [dry-run] mkdir -p $CH2/$d" || mkdir -p "$CH2/$d"
done

# =============================================================================
# 2. FIGURES — in-paper (fig01 … fig07)
# =============================================================================
divider "figures/ — in-paper"

move "$CH2/fig1_sensor_bias_AMSRE_minus_SMMR.py" \
     "$CH2/figures/fig01_sensor_bias_amsre_smmr.py"

move "$CH2/fig2_FS_MS_window_sensitivity_static_maps.py" \
     "$CH2/figures/fig02_window_sensitivity_maps.py"

move "$CH2/fig3_FS_MS_threshold_sensitivity_static_ecdfs.py" \
     "$CH2/figures/fig03_threshold_sensitivity_ecdfs.py"

move "$CH2/fig4-5_FS_MS_climatology_static_vs_dynamic.py" \
     "$CH2/figures/fig04_05_climatology_static_dynamic_maps.py"

move "$CH2/fig6_FS_MS_climatology_static_vs_dynamic_violins.py" \
     "$CH2/figures/fig06_climatology_sector_violins.py"

move "$CH2/fig7_FS_MS_trends_static_vs_dynamic.py" \
     "$CH2/figures/fig07_trends_static_dynamic.py"

# =============================================================================
# 3. FIGURES — supplements
#    NOTE: figS0X numbers are placeholders — rename to match your actual
#    supplement numbering before submitting.
# =============================================================================
divider "figures/ — supplements"

move "$CH2/fig_FS_MS_threshold_sensitivity_static_maps.py" \
     "$CH2/figures/figS01_threshold_sensitivity_maps.py"

move "$CH2/fig_FS_MS_window_sensitivity_static_ecdfs.py" \
     "$CH2/figures/figS02_window_sensitivity_ecdfs.py"

move "$CH2/fig_FS_MS_window_sensitivity_static_ecdfs_by_sector.py" \
     "$CH2/figures/figS03_window_sensitivity_ecdfs_sector.py"

move "$CH2/fig_FS_MS_anomaly_maps_static_vs_dynamic.py" \
     "$CH2/figures/figS04_anomaly_maps_pre_post.py"

move "$CH2/fig_FS_MS_anomaly_maps_static_vs_dynamic_singleyear.py" \
     "$CH2/figures/figS05_anomaly_maps_singleyear.py"

# =============================================================================
# 4. FIGURES — scratch (exploratory, not in paper)
# =============================================================================
divider "figures/scratch/"

move "$CH2/fig_phase_persistence_FS_MS_static_dynamic.py" \
     "$CH2/figures/scratch/fig_phase_persistence_static_dynamic.py"

move "$CH2/fig_SIC_seasonal_amplitude_pre2015_post2016.py" \
     "$CH2/figures/scratch/fig_sic_amplitude_pre2016_post2016.py"

move "$CH2/fig_SIC_seasonal_amplitude_pre_post.py" \
     "$CH2/figures/scratch/fig_sic_amplitude_pre2018_post2018.py"

move "$CH2/fig_SIC_persistence_melt_pre2015_post2016.py" \
     "$CH2/figures/scratch/fig_sic_persistence_melt_pre_post.py"

# =============================================================================
# 5. PROCESSING — smmr/ scripts
#    These live in smmr/ (sibling of Ch2/), so paths are relative to that.
#    Adjust SMMR if your smmr/ folder is elsewhere.
# =============================================================================
divider "processing/"

SMMR="$(dirname "$CH2")/smmr"   # assumes smmr/ is a sibling of Ch2/
PROC="$CH2/processing"

# Phase detection
move "$SMMR/fs_ms_smmr.py"                          "$PROC/compute_phase_static.py"
move "$SMMR/FS_MS_StaticWindow.py"                  "$PROC/compute_phase_static_slope.py"
move "$SMMR/compute_FS_MS_anomalies_static_dynamic.py" "$PROC/compute_phase_anomalies.py"

# Data pipeline
move "$SMMR/pipeline_SMMR.py"                       "$PROC/pipeline_update_data.py"
move "$SMMR/compute_SIE_csv.py"                     "$PROC/compute_sie_timeseries.py"
move "$SMMR/download_smmr.py"                       "$PROC/download_bootstrap_smmr.py"
move "$SMMR/merge_granules.py"                      "$PROC/merge_bootstrap_granules.py"
move "$SMMR/merge_smmr.py"                          "$PROC/merge_bootstrap_historical.py"

# Mask
move "$CH2/make_seasonal_ice_zone_mask.py"          "$PROC/compute_siz_mask.py"

# =============================================================================
# 6. UTILS
# =============================================================================
divider "utils/"

move "$CH2/ch2_fig_utils.py"   "$CH2/utils/plot_utils.py"
move "$SMMR/config.py"         "$CH2/utils/config.py"

# =============================================================================
# 7. SUMMARY
# =============================================================================
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. Run without --dry-run to apply."
else
  echo "Done. New structure:"
  find "$CH2" -type f -name "*.py" | sort | sed "s|$CH2/||"
fi
