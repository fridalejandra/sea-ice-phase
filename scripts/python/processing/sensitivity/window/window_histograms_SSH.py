import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# === USER CONFIGURATION === #
DATASETS = ["SMMR", "AMSRE"]
PHASES = ["advance", "retreat"]
WINDOWS = [3, 5, 7]
MAX_DOY = 366

# === Wrapped difference function === #
def wrapped_difference(a, b):
    raw = a - b
    return ((raw + MAX_DOY // 2) % MAX_DOY) - (MAX_DOY // 2)

# === GOOGLE DRIVE OUTPUT (RCLONE) === #
RCLONE_REMOTE = "gdrive:sea-ice-phase/results/figures/window_histograms_wrapped"

for dataset in DATASETS:
    years = range(1979, 2024) if dataset == "SMMR" else range(2012, 2024)
    base_dir = f"/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/{dataset}_phase/{dataset}_window_comparison"

    for phase in PHASES:
        print(f"\n=== Processing {dataset} - {phase} ===")

        diff_3_stack = []
        diff_7_stack = []

        for year in years:
            try:
                f3 = os.path.join(base_dir, f"diff_{phase}_3minus5_{year}.nc")
                f7 = os.path.join(base_dir, f"diff_{phase}_7minus5_{year}.nc")

                ds3 = xr.open_dataset(f3)
                ds7 = xr.open_dataset(f7)

                diff3_raw = ds3[f"diff_{phase}_3minus5"].values
                diff7_raw = ds7[f"diff_{phase}_7minus5"].values

                diff3_wrapped = wrapped_difference(diff3_raw, 0)
                diff7_wrapped = wrapped_difference(diff7_raw, 0)

                diff_3_stack.append(diff3_wrapped)
                diff_7_stack.append(diff7_wrapped)

            except FileNotFoundError:
                print(f"Skipping year {year} — file not found.")
            except Exception as e:
                print(f"Error processing {year}: {e}")

        # === Flatten and clean === #
        all_diff3 = np.array(diff_3_stack).flatten()
        all_diff7 = np.array(diff_7_stack).flatten()

        valid_diff3 = all_diff3[~np.isnan(all_diff3)]
        valid_diff7 = all_diff7[~np.isnan(all_diff7)]

        # === Plotting === #
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        axes[0].hist(valid_diff3, bins=60, color="steelblue", edgecolor="black")
        axes[0].axvline(0, linestyle="--", color="black", linewidth=1)
        axes[0].set_title("3-Day Minus 5-Day (Wrapped)")
        axes[0].set_xlabel("Timing Difference (days)")
        axes[0].set_ylabel("Number of Pixels")

        axes[1].hist(valid_diff7, bins=60, color="tomato", edgecolor="black")
        axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
        axes[1].set_title("7-Day Minus 5-Day (Wrapped)")
        axes[1].set_xlabel("Timing Difference (days)")

        fig.suptitle(f"{phase.capitalize()} Timing Difference — {dataset}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # === Save and rclone === #
        fig_name = f"histogram_wrapped_window_diff_{phase}_{dataset}.png"
        local_path = f"/tmp/{fig_name}"
        plt.savefig(local_path, dpi=300)
        plt.close()

        os.system(f"rclone copy '{local_path}' '{RCLONE_REMOTE}'")
        print(f"✅ Uploaded: {fig_name}")
