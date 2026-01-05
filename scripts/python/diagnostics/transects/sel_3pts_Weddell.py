import pandas as pd
from pathlib import Path

transect_csv = Path(
    "/user/geog/falejandraperez/sea-ice-phase/results/diagnostics/"
    "edge_distance/bootstrap_ws/seasonal/"
    "transect_Weddell_meridional_lon-30.0_GROWTH_MAMJ.csv"
)

df = pd.read_csv(transect_csv)

targets = {
    "MIZ_50km": 50.0,
    "INNER_PACK_500km": 500.0,
}

points = {}

# MIZ and inner pack
for name, d0 in targets.items():
    idx = (df["dist_to_edge_km"] - d0).abs().idxmin()
    points[name] = df.loc[idx]

# Southernmost
idx_south = df["lat"].idxmin()
points["SOUTHERN_TRANSECT"] = df.loc[idx_south]

points_df = pd.DataFrame(points).T
out_csv = transect_csv.parent / "selected_transect_points_Weddell_lon-30.csv"
points_df.to_csv(out_csv, index=True)

print(points_df[["lat", "lon", "dist_to_edge_km", "x_idx", "y_idx"]])
print(f"\nSaved point metadata to: {out_csv}")
