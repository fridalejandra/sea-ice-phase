import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# === USER CONFIGURATION === #
DATASETS = ["SMMR", "AMSRE"]
PHASES = ["advance", "retreat"]
WINDOWS = [3, 5, 7]
RCLONE_REMOTE = "gdrive:sea-ice-phase/results/figures/window_histograms"

for dataset in DATASETS:
    years = range(1979, 2024) if dataset == "SMMR" else range(2012, 2024)
    base_dir = f"/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/{dataset}_phase/{dataset}_window_comparison"

    for phase in PHASES:
        print(f"\n=== Processing {dataset} - {phase} ===")

        diff_3_stack = []
        diff_7_stack = []

        for year in years:
            try:
                f3 = os.path.join(base_dir, phase, f"diff_{phase}_3minus5_{year}.nc")
                f7 = os.path.join(base_dir, phase, f"diff_{phase}_7minus5_{year}.nc")

                ds3 = xr.open_dataset(f3)
                ds7 = xr.open_dataset(f7)

                diff3 = ds3[f"diff_{phase}_3minus5"]
                diff7 = ds7[f"diff_{phase}_7minus5"]

                diff_3_stack.append(diff3)
                diff_7_stack.append(diff7)

            except FileNotFoundError:
                print(f"Skipping year {year} — file not found.")
            except Exception as e:
                print(f"Error processing {year}: {e}")

        # === Flatten and mask === #
        all_diff3 = xr.concat(diff_3_stack, dim="time").values.flatten()
        all_diff7 = xr.concat(diff_7_stack, dim="time").values.flatten()

        valid_diff3 = all_diff3[~np.isnan(all_diff3)]
        valid_diff7 = all_diff7[~np.isnan(all_diff7)]

        # === Plotting === #
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        axes[0].hist(valid_diff3, bins=60, color="steelblue", edgecolor="black")
        axes[0].axvline(0, linestyle="--", color="black", linewidth=1)
        axes[0].set_title("3-Day Minus 5-Day")
        axes[0].set_xlabel("Timing Difference (days)")
        axes[0].set_ylabel("Number of Pixels")

        axes[1].hist(valid_diff7, bins=60, color="tomato", edgecolor="black")
        axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
        axes[1].set_title("7-Day Minus 5-Day")
        axes[1].set_xlabel("Timing Difference (days)")

        fig.suptitle(f"{phase.capitalize()} Timing Difference — {dataset}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # === Save to temp + Upload to Google Drive === #
        fig_name = f"histogram_window_diff_{phase}_{dataset}.png"
        temp_path = f"/tmp/{fig_name}"
        plt.savefig(temp_path, dpi=300)
        plt.close()

        # Upload using rclone
        os.system(f"rclone copy '{temp_path}' '{RCLONE_REMOTE}'")
        print(f"✅ Uploaded: {fig_name}")
