import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Load the summary CSV exported from R
summary_path = "/Users/fridaperez/Desktop/summary_phase_metrics.csv"
summary = pd.read_csv(summary_path)

# Helper function to calculate rolling mean anomalies and std deviation
def rolling_mean_anomaly_corrected(series, window=6):
    rolling = uniform_filter1d(series, size=window, mode='nearest')
    anomaly = rolling - np.nanmean(series)
    std_dev = np.nanstd(series)
    return rolling, anomaly, std_dev

# Plot function
def plot_anomaly(years, anomalies, std, title, ylabel, color):
    plt.figure(figsize=(10, 4))
    plt.plot(years, anomalies, color=color, label="6-Year Anomaly")
    plt.fill_between(years, anomalies - std, anomalies + std, color=color, alpha=0.2, label="±1 Std Dev")
    plt.axhline(0, linestyle="dotted", color="black")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Sort and trim to exclude first and last year
summary = summary.sort_values("Year")
filtered = summary[(summary["Year"] > summary["Year"].min()) & (summary["Year"] < summary["Year"].max())]

# Compute corrected anomalies for durations and rates
filtered["Freeze_Duration_6yr"], filtered["Anom_Freeze_Duration_6yr"], std_freeze = rolling_mean_anomaly_corrected(filtered["Freezing_Duration"])
filtered["Melt_Duration_6yr"], filtered["Anom_Melt_Duration_6yr"], std_melt = rolling_mean_anomaly_corrected(filtered["Melting_Duration"])
filtered["Rate_Advance_6yr"], filtered["Anom_Rate_Advance_6yr"], std_adv = rolling_mean_anomaly_corrected(filtered["Rate_Advance"])
filtered["Rate_Retreat_6yr"], filtered["Anom_Rate_Retreat_6yr"], std_ret = rolling_mean_anomaly_corrected(filtered["Rate_Retreat"])

# Generate plots
plot_anomaly(filtered["Year"], filtered["Anom_Freeze_Duration_6yr"], std_freeze,
             "6-Year Mean Anomaly: Freeze Duration", "Anomaly (days)", "skyblue")
plot_anomaly(filtered["Year"], filtered["Anom_Melt_Duration_6yr"], std_melt,
             "6-Year Mean Anomaly: Melt Duration", "Anomaly (days)", "darkorange")
plot_anomaly(filtered["Year"], filtered["Anom_Rate_Retreat_6yr"], std_ret,
             "6-Year Mean Anomaly: Rate of Retreat", "Anomaly (10⁶ km²/day)", "tomato")
plot_anomaly(filtered["Year"], filtered["Anom_Rate_Advance_6yr"], std_adv,
             "6-Year Mean Anomaly: Rate of Advance", "Anomaly (10⁶ km²/day)", "navy")
