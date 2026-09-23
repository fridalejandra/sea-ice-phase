#!/bin/bash
#SBATCH --job-name=ch3_figures
#SBATCH --output=logs/ch3_figures_%j.out
#SBATCH --error=logs/ch3_figures_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal

# ============================================================
# Ch3 Figure Runner — SLURM
# Submit from figures directory:
#   sbatch run_ch3_figures_cluster.sh
#
# Or run interactively on cluster:
#   bash run_ch3_figures_cluster.sh
# ============================================================

set -e

# ── Environment ──────────────────────────────────────────────
# Adjust conda env name if needed
source ~/.bashrc
conda activate seaice

# ── Paths ────────────────────────────────────────────────────
FIGURES_DIR=~/Research/repos/sea-ice-phase/scripts/python/plotting/Ch3/figures
LOG_DIR=$FIGURES_DIR/logs

mkdir -p $LOG_DIR
cd $FIGURES_DIR

echo "============================================"
echo "  Ch3 Figure Run"
echo "  Start: $(date)"
echo "  Dir:   $FIGURES_DIR"
echo "============================================"

# ── Helper: run one script with timing and error capture ─────
run_fig() {
    local script=$1
    local fig=$2
    echo ""
    echo "── $fig ──────────────────────────────────────"
    echo "   Script: $script"
    echo "   Start:  $(date +%H:%M:%S)"
    
    if python "$script" 2>&1 | tee -a "$LOG_DIR/${script%.py}.log"; then
        echo "   Done:   $(date +%H:%M:%S) ✓"
    else
        echo "   FAILED: $(date +%H:%M:%S) ✗ — see $LOG_DIR/${script%.py}.log"
        # Continue to next figure rather than aborting whole run
    fi
}

# ── Main figures ─────────────────────────────────────────────
run_fig fig01_conflation.py              "Fig  1 — Conflation schematic"
run_fig fig02_sector_map.py              "Fig  2 — Sector map"
run_fig fig03_phase_amp_independence.py  "Fig  3 — Phase-amplitude independence"
run_fig fig04_dot_timeline.py            "Fig  4 — Dot timeline"
run_fig fig05_phase_amp_scatter.py       "Fig  5 — Phase-amplitude scatter"
run_fig fig06_phase_timeseries.py        "Fig  6 — Phase timeseries"
run_fig fig07_amplitude_timeseries.py    "Fig  7 — Amplitude timeseries"
run_fig fig06_rolling_variance.py        "Fig  8 — Rolling variance"
run_fig fig09_correlation_heatmap.py     "Fig  9 — Correlation heatmap"
run_fig fig10_shoulder_season_clustermap.py "Fig 10 — Shoulder season"
run_fig fig12_rolling_window_top3.py     "Fig 11 — Rolling window top 3"
run_fig fig12_rolling_phase_amp_sie.py   "Fig 12 — Rolling phase/amp vs SIE"

# ── Supplement ───────────────────────────────────────────────
run_fig figS01_fitted_vs_observed.py     "Fig S1/S2 — Fitted vs observed"

echo ""
echo "============================================"
echo "  All done: $(date)"
echo "============================================"

# ── Check which figures were produced ────────────────────────
echo ""
echo "Output files in figures directory:"
ls -lh *.png 2>/dev/null | awk '{print $5, $9}' | sort -k2



