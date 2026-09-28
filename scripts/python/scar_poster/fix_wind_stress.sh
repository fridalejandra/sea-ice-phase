#!/usr/bin/env bash
#
# fix_wind_stress.sh
#
# Fixes the wind stress deaccumulation bug in one shot:
#   1. Patches ACCUM_SECONDS 86400 -> 3600 in compute_gridded_sic_wind_diff.py
#   2. Patches regrid_wind_to_ease.py: no differencing, divide by 3600
#   3. Deletes the buggy regridded outputs
#   4. Reruns the regrid (weights cached, so fast-ish)
#   5. Regenerates the SIC+wind overlay figure
#
# Run from scar_poster/ root:
#   bash fix_wind_stress.sh
#
set -euo pipefail

cd /user/geog/falejandraperez/sea-ice-phase/scripts/python/scar_poster

echo "=== 1. Patch compute_gridded_sic_wind_diff.py ==="
if [ -f pipeline/compute_gridded_sic_wind_diff.py ]; then
    sed -i 's/ACCUM_SECONDS = 86400/ACCUM_SECONDS = 3600/' \
        pipeline/compute_gridded_sic_wind_diff.py
    grep "ACCUM_SECONDS" pipeline/compute_gridded_sic_wind_diff.py
    echo "  patched."
else
    echo "  [skip] pipeline/compute_gridded_sic_wind_diff.py not found"
fi

echo
echo "=== 2. Patch regrid_wind_to_ease.py ==="
# find it whether it's in root or pipeline/
REGRID=""
for candidate in regrid_wind_to_ease.py pipeline/regrid_wind_to_ease.py; do
    if [ -f "$candidate" ]; then REGRID="$candidate"; break; fi
done
if [ -z "$REGRID" ]; then
    echo "  [STOP] regrid_wind_to_ease.py not found in root or pipeline/"
    exit 1
fi
echo "  found: $REGRID"

# flip the flag and the constant
sed -i 's/DEACCUMULATE = True/DEACCUMULATE = False/' "$REGRID"
sed -i 's/ACCUM_SECONDS = 86400\.0/ACCUM_SECONDS = 3600.0/' "$REGRID"

# replace the deaccumulation block with an if/else that divides when not
# differencing. Uses python for the multi-line edit (sed is fragile here).
python - "$REGRID" << 'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = """    if DEACCUMULATE:
        # difference consecutive timesteps to get instantaneous stress
        tau_x = tau_x.diff(dim=TIME_COORD) / ACCUM_SECONDS
        tau_y = tau_y.diff(dim=TIME_COORD) / ACCUM_SECONDS"""

new = """    if DEACCUMULATE:
        # difference consecutive timesteps to get instantaneous stress
        tau_x = tau_x.diff(dim=TIME_COORD) / ACCUM_SECONDS
        tau_y = tau_y.diff(dim=TIME_COORD) / ACCUM_SECONDS
    else:
        # values are per-hour accumulations delivered as daily means:
        # divide by 3600 s to get stress in Pa
        tau_x = tau_x / ACCUM_SECONDS
        tau_y = tau_y / ACCUM_SECONDS"""

if "else:\n        # values are per-hour accumulations" in src:
    print("  already patched, skipping block edit")
elif old in src:
    src = src.replace(old, new)
    with open(path, "w") as f:
        f.write(src)
    print("  deaccumulation block patched")
else:
    print("  [WARN] expected block not found -- check the file manually")
    sys.exit(1)
PYEOF

grep -A1 "DEACCUMULATE =" "$REGRID" | head -2

echo
echo "=== 3. Delete buggy outputs ==="
rm -fv wind_stress_on_ease_sh.nc wind_stress_curl_on_ease_sh.nc

echo
echo "=== 4. Rerun regrid (weights cached) ==="
python "$REGRID"

echo
echo "=== 5. Regenerate overlay figure ==="
python fig_sic_wind_overlay.py

echo
echo "=== DONE ==="
echo "Check the new fig_sic_wind_overlay.png: vectors should now show"
echo "coherent eastward (westerly) intensification in the 50-65S band."
echo "If you keep the wind stress / curl maps from plot_monthly_maps.py,"
echo "rerun that too -- its inputs were just regenerated."