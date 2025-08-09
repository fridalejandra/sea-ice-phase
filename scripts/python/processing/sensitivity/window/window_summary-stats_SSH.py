import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION === #
DATASETS = ["SMMR", "AMSRE"]
PHASES = ["advance", "retreat"]
WINDOWS = [3, 5, 7]
THRESHOLD = 5  # Days
RCLONE_REMOTE = "gdrive:sea-ice-phase/results/figures/window_summary"
LOCAL_TMPDIR = "/tmp/seaice_figs"  # Temporary save location before rclone
os.makedirs(LOCAL_TMPDIR, exist_ok=True)

for dataset in DATASETS:
    years = range(1979, 2024) if dataset == "SMMR" else range(2012, 2024)
    base_dir = f"/user/geog/falejandraperez/sea-ice-phase/results/sensitivity/{dataset}_phase/{dataset}_window_comparison"

    for phase in PHASES:
        print(f"\n=== Processing {dataset} - {phase} ===")
        means_3, stds_3, perc_3 = [], [], []
        means_7, stds_7, perc_7 = [], [], []

        for year in years:
            try:
                f3 = os.path.join(base_dir, f"diff_{phase}_3minus5_{year}.nc")
                f7 = os.path.join(base_dir, f"diff_{phase}_7minus5_{year}.nc")

                diff3 = xr.open_dataset(f3)[f"diff_{phase}_3minus5"]
                diff7 = xr.open_dataset(f7)[f"diff_{phase}_7minus5"]

                valid3 = diff3.values.flatten()
                valid7 = diff7.values.flatten()
                valid3 = valid3[~np.isnan(valid3)]
                valid7 = valid7[~np.isnan(valid7)]

                means_3.append(np.mean(valid3))
                stds_3.append(np.std(valid3))
                perc_3.append(np.sum(np.abs(valid3) > THRESHOLD) / len(valid3) * 100)

                means_7.append(np.mean(valid7))
                stds_7.append(np.std(valid7))
                perc_7.append(np.sum(np.abs(valid7) > THRESHOLD) / len(valid7) * 100)

            except FileNotFoundError:
                print(f"Skipping year {year}")
            except Exception as e:
                print(f"Error in year {year}: {e}")

        # === PLOT BAR CHARTS === #
        labels = list(years[:len(means_3)])  # Trim if any missing
        x = np.arange(len(labels))
        width = 0.35

        def barplot(values1, values2, ylabel, title, filename):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(x - width/2, values1, width, label='3–5 Day', color='steelblue')
            ax.bar(x + width/2, values2, width, label='7–5 Day', color='tomato')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            save_path = os.path.join(LOCAL_TMPDIR, filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
            os.system(f"rclone copy '{save_path}' '{RCLONE_REMOTE}'")
            print(f"✅ Uploaded: {filename}")

        # Mean
        barplot(means_3, means_7,
                ylabel="Mean Timing Difference (days)",
                title=f"{phase.capitalize()} Timing Mean Diff — {dataset}",
                filename=f"mean_diff_window_{phase}_{dataset}.png")

        # Std
        barplot(stds_3, stds_7,
                ylabel="Standard Deviation (days)",
                title=f"{phase.capitalize()} Timing Std Dev — {dataset}",
                filename=f"std_diff_window_{phase}_{dataset}.png")

        # % Pixels > 5-day diff
        barplot(perc_3, perc_7,
                ylabel="% of Pixels with >5-Day Diff",
                title=f"{phase.capitalize()} % Pixels >5-Day — {dataset}",
                filename=f"pct_over5_window_{phase}_{dataset}.png")
