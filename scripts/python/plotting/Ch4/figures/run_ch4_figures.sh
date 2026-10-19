#!/bin/bash
# Run on the cluster from scripts/python/processing/ERA5_Reanalysis/ (ch4_style.py,
# ch4_figstyle.py, make_ch4_figures.py must sit together). Then push to Drive.
set -e
conda activate era5_regrid 2>/dev/null || source activate era5_regrid
python make_ch4_figures.py                     # all 6, both modes (fig02 is manual; see printout)
FIG=/user/geog/falejandraperez/sea-ice-phase/results/ch4/figures
rclone copy "$FIG/manuscript" 'gdrive:My Drive/sea-ice-phase/results/Ch4_Figures/manuscript/' -P
rclone copy "$FIG/slides"     'gdrive:My Drive/sea-ice-phase/results/Ch4_Figures/slides/'     -P
rclone ls 'gdrive:My Drive/sea-ice-phase/results/Ch4_Figures/'
