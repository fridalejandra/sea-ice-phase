#!/usr/bin/env python3
"""
Select 3 objective points along the Weddell meridional transect (near 30°W)
using climatological distance-to-edge targets.

Targets:
- MIZ_50km:          dist_to_edge_km closest to 50
- INNER_PACK_500km:  dist_to_edge_km closest to 500
- DEEP_PACK_1000km:  dist_to_edge_km closest to 1000

Writes a CSV used downstream by the daily extraction + plotting scripts.
"""

from pathlib import Path
import pandas as pd


def main():
    transect_csv = Path(
        "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/"
        "edge_distance/bootstrap_ws/seasonal/"
        "transect_Weddell_meridional_lon-30.0_GROWTH_MAMJ.csv"
    )

    out_csv = Path(
        "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/"
        "edge_distance/bootstrap_ws/seasonal/"
        "selected_transect_points_Weddell_lon-30.csv"
    )

    print(f"[info] reading transect: {transect_csv}", flush=True)
    df = pd.read_csv(transect_csv)

    # Basic checks
    needed = {"lat", "lon", "dist_to_edge_km"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Transect CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Handle index columns from earlier scripts (optional)
    # If your transect file already contains x_idx/y_idx, we keep them.
    for col in ["x_idx", "y_idx"]:
        if col not in df.columns:
            # If you used different names, you should add them upstream.
            print(f"[warn] {col} not found in transect CSV. Downstream extraction requires x_idx/y_idx.", flush=True)

    targets = {
        "MIZ_50km": 50.0,
        "INNER_PACK_500km": 500.0,
        "DEEP_PACK_1000km": 1000.0,
    }

    picks = []
    used_rows = set()

    for name, d0 in targets.items():
        # choose closest unused row
        s = (df["dist_to_edge_km"] - d0).abs()
        # if we already used the closest row for a previous target, take next closest
        for idx in s.sort_values().index:
            if idx not in used_rows:
                used_rows.add(idx)
                row = df.loc[idx].copy()
                row["point"] = name
                picks.append(row)
                break

    out = pd.DataFrame(picks)

    # Reorder columns to be tidy
    front = ["point", "lat", "lon", "dist_to_edge_km"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    print("[info] selected points:", flush=True)
    show_cols = [c for c in ["point", "lat", "lon", "dist_to_edge_km", "x_idx", "y_idx"] if c in out.columns]
    print(out[show_cols].to_string(index=False), flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[info] wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
