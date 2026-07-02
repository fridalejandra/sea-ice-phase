#!/usr/bin/env python3
"""
FigS_weddell_MS_crossing_freq_by_year.py

Year-by-year MS threshold crossing frequency in the Weddell sector,
showing the transition from stable pack interior (pre-2016) to 
a more variable regime (post-2016).
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import sys

PROJECT_ROOT = Path('/user/geog/falejandraperez/sea-ice-phase')
sys.path.insert(0, str(PROJECT_ROOT / 'scripts/python/plotting/Ch2'))
from utils.plot_utils import get_fig_path, save_and_upload

ds = xr.open_dataset(PROJECT_ROOT / 'data/transition_metrics/SMMR/crossing_freq_MS_thr15.nc')
cf = ds['crossing_freq']

EXTENT = [-60, -20, -80, -55]
PRE_END = 2015
POST_START = 2016

years = cf.year.values
ncols = 6
nrows = int(np.ceil(len(years) / ncols))

fig = plt.figure(figsize=(18, nrows * 3))
proj = ccrs.SouthPolarStereo()

for i, yr in enumerate(years):
    ax = fig.add_subplot(nrows, ncols, i+1, projection=proj)
    ax.set_extent(EXTENT, ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='0.8', zorder=2)
    ax.pcolormesh(cf.x, cf.y, cf.sel(year=yr),
                  transform=proj, cmap='RdBu_r',
                  vmin=0, vmax=5, shading='auto')
    
    # highlight post-2016 titles
    color = '#d73027' if yr >= POST_START else 'black'
    ax.set_title(str(yr), fontsize=7, fontweight='bold', color=color)

fig.suptitle('MS threshold crossing frequency — Weddell sector (1979–2024)\n'
             'Red titles = post-2016', fontsize=11, y=1.01)

# shared colorbar
cax = fig.add_axes([0.15, 0.02, 0.7, 0.01])
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 5))
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('Crossings per season', fontsize=9)
cb.outline.set_visible(False)

fig.tight_layout(rect=[0, 0.04, 1, 1])

out_path = get_fig_path(PROJECT_ROOT, subfolder='', 
                        fig_name='FigS_weddell_MS_crossing_freq_by_year.png')
save_and_upload(fig, out_path, 
                remote_root='gdrive:sea-ice-phase/results/Ch2_Figures',
                remote_subdir='')
ds.close()
