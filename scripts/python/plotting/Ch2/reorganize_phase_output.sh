#!/usr/bin/env bash
# =============================================================================
# reorganize_phase_output.sh
#
# Moves phase detection output from results/ into the clean data/ structure:
#
#   FROM (results/):
#     results/SMMR_phase/static/FS_thr15_k5/FS_YYYY.nc
#     results/SMMR_phase/static/MS_thr15_k5/MS_YYYY.nc
#     results/SMMR_phase/FS_thr15_k5/FS_YYYY.nc   (old flat structure)
#
#   TO (data/):
#     data/SMMR_phase/static/thr15_k5/FS/FS_YYYY.nc
#     data/SMMR_phase/static/thr15_k5/MS/MS_YYYY.nc
#     data/AMSRE_phase/static/thr15_k5/FS/FS_YYYY.nc
#     data/SMMR_phase/dynamic/k5_q70/FS/FS_YYYY.nc
#
# Run AFTER compute_phase_dates.py finishes.
# Pass --dry-run to preview without moving anything.
# =============================================================================

set -euo pipefail

ROOT="/user/geog/falejandraperez/sea-ice-phase"
RESULTS="$ROOT/results"
DATA="$ROOT/data"

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

[[ $DRY_RUN -eq 1 ]] && echo "[DRY RUN]"

# ── helpers ──────────────────────────────────────────────────────────────────
move_files() {
  local src="$1" dst="$2"
  if [[ ! -d "$src" ]]; then
    echo "  [skip] $src not found"
    return
  fi
  local count
  count=$(find "$src" -name "*.nc" | wc -l)
  if [[ $count -eq 0 ]]; then
    echo "  [skip] $src empty"
    return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry] mv $count files: ${src#$ROOT/} → ${dst#$ROOT/}"
    return
  fi
  mkdir -p "$dst"
  mv "$src"/*.nc "$dst"/
  echo "  mv $count files: ${src#$ROOT/} → ${dst#$ROOT/}"
}

divider() { echo ""; echo "── $1 ──"; }

# =============================================================================
# 1. SMMR — new structure (results/SMMR_phase/static/FS_thr15_k5/)
# =============================================================================
divider "SMMR static — new structure"

for phase in FS MS ME; do
  for combo in "$RESULTS/SMMR_phase/static/${phase}_thr"*; do
    [[ ! -d "$combo" ]] && continue
    dirname=$(basename "$combo")
    # extract thr and k from e.g. FS_thr15_k5
    params="${dirname#${phase}_}"   # → thr15_k5
    dst="$DATA/SMMR_phase/static/$params/$phase"
    move_files "$combo" "$dst"
  done
done

# =============================================================================
# 2. SMMR — old flat structure (results/SMMR_phase/FS_thr15_k5/)
# =============================================================================
divider "SMMR static — old flat structure"

for phase in FS MS ME; do
  for combo in "$RESULTS/SMMR_phase/${phase}_thr"*; do
    [[ ! -d "$combo" ]] && continue
    dirname=$(basename "$combo")
    params="${dirname#${phase}_}"   # → thr15_k5
    dst="$DATA/SMMR_phase/static/$params/$phase"
    move_files "$combo" "$dst"
  done
done

# =============================================================================
# 3. SMMR — dynamic (results/SMMR_phase/dynamic/quantile_k5/FS/p7/)
# =============================================================================
divider "SMMR dynamic"

for phase in FS MS; do
  src="$RESULTS/SMMR_phase/dynamic/quantile_k5/$phase/p7"
  dst="$DATA/SMMR_phase/dynamic/k5_q70/$phase"
  move_files "$src" "$dst"
done

# =============================================================================
# 4. AMSRE — new structure
# =============================================================================
divider "AMSRE static — new structure"

for phase in FS MS; do
  for combo in "$RESULTS/AMSRE_phase/static/${phase}_thr"*; do
    [[ ! -d "$combo" ]] && continue
    dirname=$(basename "$combo")
    params="${dirname#${phase}_}"
    dst="$DATA/AMSRE_phase/static/$params/$phase"
    move_files "$combo" "$dst"
  done
done

# =============================================================================
# 5. AMSRE — dynamic
# =============================================================================
divider "AMSRE dynamic"

for phase in FS MS; do
  src="$RESULTS/AMSRE_phase/dynamic/quantile_k5/$phase/p7"
  dst="$DATA/AMSRE_phase/dynamic/k5_q70/$phase"
  move_files "$src" "$dst"
done

# =============================================================================
# 6. SUMMARY
# =============================================================================
echo ""
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete. Run without --dry-run to apply."
else
  echo "Done. New structure:"
  find "$DATA/SMMR_phase" "$DATA/AMSRE_phase" -type d 2>/dev/null | sort | sed "s|$ROOT/||"
fi