#!/bin/bash
cd /user/geog/falejandraperez/sea-ice-phase/scripts/python/plotting/Ch2/processing

# name : Q_FS : Q_MS : slope_min
RUNS="
q60_s02:0.60:0.40:0.02
q80_s02:0.80:0.20:0.02
q70_s01:0.70:0.30:0.01
q70_s04:0.70:0.30:0.04
"

for spec in $RUNS; do
  NAME=$(echo $spec | cut -d: -f1)
  QFS=$(echo  $spec | cut -d: -f2)
  QMS=$(echo  $spec | cut -d: -f3)
  SLP=$(echo  $spec | cut -d: -f4)
  SCRIPT=sweep_${NAME}.py

  cp compute_phase_dates_v2_yearsfix.py $SCRIPT
  sed -i "s|^DYN_Q_FS       = 0.70|DYN_Q_FS       = ${QFS}|" $SCRIPT
  sed -i "s|^DYN_Q_MS       = 0.30|DYN_Q_MS       = ${QMS}|" $SCRIPT
  sed -i "s|^DYN_SLOPE_MIN  = 0.02|DYN_SLOPE_MIN  = ${SLP}|" $SCRIPT
  sed -i "s|f\"{varname}_{year}.nc\"|f\"sweep_${NAME}_{varname}_{year}.nc\"|" $SCRIPT

  echo "===== SWEEP RUN ${NAME}: QFS=${QFS} QMS=${QMS} SLOPE=${SLP} ====="
  python $SCRIPT --method dynamic
  echo "===== SWEEP RUN ${NAME} DONE (exit $?) ====="
done
echo "ALL SWEEP RUNS COMPLETE"
