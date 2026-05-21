#!/usr/bin/env bash
# =============================================================================
# reorganize_ch2_processing.sh
#
# Phase 2 cleanup — moves the scattered processing/sensitivity/ scripts into
# the Ch2 structure and removes old _SSH scratch files and duplicate folders.
#
# Final structure:
#
#   scripts/python/
#   ├── processing/
#   │   ├── smmr/        ← ONLY download + merge + pipeline (data plumbing)
#   │   ├── amsre/       ← ONLY download + merge + convert  (data plumbing)
#   │   └── ERA5_Reanalysis/  (untouched)
#   │
#   └── plotting/Ch2/
#       ├── figures/     (already done in phase 1)
#       ├── processing/
#       │   ├── compute_phase_static.py
#       │   ├── compute_phase_static_slope.py
#       │   ├── compute_phase_dynamic.py          ← was run_dynamic_thresholds_staticSlope.py
#       │   ├── compute_phase_anomalies.py
#       │   ├── compute_phase_amsre.py             ← was advance_retreat_amsre.py
#       │   ├── compute_sie_timeseries.py
#       │   └── compute_siz_mask.py
#       └── utils/
#           ├── plot_utils.py
#           └── config.py
#
# Run from the repo root:
#   bash scripts/python/plotting/Ch2/reorganize_ch2_processing.sh --dry-run
#   bash scripts/python/plotting/Ch2/reorganize_ch2_processing.sh
# =============================================================================

set -euo pipefail

ROOT="/user/geog/falejandraperez/sea-ice-phase"
CH2="$ROOT/scripts/python/plotting/Ch2"
PROC="$ROOT/scripts/python/processing"

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
    printf "  [dry-run] mv %-70s →  %s\n" "${src#$ROOT/}" "${dst#$ROOT/}"
  else
    mv "$src" "$dst"
    printf "  mv %-70s →  %s\n" "${src#$ROOT/}" "${dst#$ROOT/}"
  fi
}

remove() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "  [SKIP – not found] $f"
    return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] rm ${f#$ROOT/}"
  else
    rm "$f"
    echo "  rm ${f#$ROOT/}"
  fi
}

remove_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "  [SKIP – not found] $d"
    return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] rmdir ${d#$ROOT/}"
  else
    rmdir --ignore-fail-on-non-empty "$d" 2>/dev/null || true
    echo "  rmdir ${d#$ROOT/}"
  fi
}

divider() { echo ""; echo "── $1 ──"; }

# ── confirmation ──────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 && $FORCE -eq 0 ]]; then
  echo "This will reorganize scripts under:"
  echo "  $ROOT/scripts/python/"
  echo ""
  read -r -p "Continue? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

[[ $DRY_RUN -eq 1 ]] && echo "[DRY RUN — no files will be moved or deleted]"
echo ""

# =============================================================================
# 1. MOVE — dynamic detection (the missing script)
# =============================================================================
divider "processing/ → Ch2/processing/ : phase detection"

move "$PROC/sensitivity/threshold/run_dynamic_thresholds_staticSlope.py" \
     "$CH2/processing/compute_phase_dynamic.py"

# AMSR-E static detection (was never in Ch2 before)
move "$PROC/amsre/advance_retreat_amsre.py" \
     "$CH2/processing/compute_phase_amsre.py"

# =============================================================================
# 2. MOVE — duplicate smmr scripts (processing/smmr/ has originals;
#    Ch2/processing/ got copies in phase 1 — keep Ch2 versions, remove smmr/ extras)
#    We only keep data-plumbing scripts in processing/smmr/
# =============================================================================
divider "processing/smmr/ — remove phase-detection duplicates (kept in Ch2)"

# These now live in Ch2/processing/ — remove from processing/smmr/
remove "$PROC/smmr/fs_ms_smmr.py"
remove "$PROC/smmr/FS_MS_StaticWindow.py"
remove "$PROC/smmr/compute_FS_MS_anomalies_static_dynamic.py"

# config and plot utils already moved to Ch2/utils/ in phase 1
remove "$PROC/smmr/config.py"

# Data-plumbing scripts stay in processing/smmr/ — do not touch:
#   pipeline_SMMR.py, download_smmr.py, merge_granules.py,
#   merge_smmr.py, compute_SIE_csv.py

# =============================================================================
# 3. DELETE — old _SSH scratch files (superseded by numbered Ch2 figure scripts)
# =============================================================================
divider "sensitivity/ — delete _SSH scratch files"

remove "$PROC/sensitivity/window/window_histograms_SSH.py"
remove "$PROC/sensitivity/window/window_maps_SSH.py"
remove "$PROC/sensitivity/window/window_summary-stats_SSH.py"
remove "$PROC/sensitivity/sensor/compare_sensors_AMSRE_SMMR_SSH.py"
remove "$PROC/sensitivity/sensor/Pearson_Corr_Sensors_SSH.py"

# =============================================================================
# 4. DELETE — sensitivity scripts superseded by Ch2 figure scripts
#    (compare_dynamic_vs_static and threshold/window sensitivity scripts
#     are fully replaced by fig02, fig03, figS01-S03)
# =============================================================================
divider "sensitivity/ — delete scripts superseded by Ch2 figures"

remove "$PROC/sensitivity/threshold/compare_dynamic_vs_static.py"
remove "$PROC/sensitivity/threshold/threshold_sensitivity_histograms.py"
remove "$PROC/sensitivity/threshold/threshold_sensitivity_maps.py"
remove "$PROC/sensitivity/window/window_sensitivity_MS_FS-static.py"
remove "$PROC/sensitivity/window/window_sensitivity_MS_FS.py"
remove "$PROC/sensitivity/window/window_sensitivity_maps.py"

# =============================================================================
# 5. CLEAN UP — remove now-empty sensitivity subdirectories
# =============================================================================
divider "sensitivity/ — remove empty dirs"

remove_dir "$PROC/sensitivity/threshold"
remove_dir "$PROC/sensitivity/window"
remove_dir "$PROC/sensitivity/sensor"
remove_dir "$PROC/sensitivity"

# =============================================================================
# 6. SUMMARY
# =============================================================================
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. Run without --dry-run to apply."
else
  echo "Done. Ch2 processing scripts:"
  find "$CH2/processing" -name "*.py" | sort | sed "s|$ROOT/||"
  echo ""
  echo "Remaining in processing/smmr/ (data plumbing only):"
  find "$PROC/smmr" -name "*.py" | sort | sed "s|$ROOT/||"
  echo ""
  echo "Remaining in processing/amsre/ (data plumbing only):"
  find "$PROC/amsre" -name "*.py" | sort | sed "s|$ROOT/||"
fi
