import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import linregress
import pandas as pd
import numpy as np


# Load your summary again
file_path = "/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv"
df = pd.read_csv(file_path, skiprows=[1])
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df['Extent'] = pd.to_numeric(df['Extent'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['DOY'] = df['Date'].dt.dayofyear


# === Calculate DOY of retreat (min) and advance (max) ===
timing = df.groupby('Year').agg(
    DOY_Retreat=('DOY', lambda x: x[df.loc[x.index, 'Extent'].idxmin()]),
    DOY_Advance=('DOY', lambda x: x[df.loc[x.index, 'Extent'].idxmax()])
).reset_index()

timing = timing[(timing['Year'] > 1978) & (timing['Year'] < 2023)].reset_index(drop=True)

# === Rolling slope calculation ===
window_size = 6
years = []
retreat_slopes = []
advance_slopes = []

for i in range(len(timing) - window_size + 1):
    window = timing.iloc[i:i + window_size]
    years.append(window['Year'].iloc[-1])
    retreat_slopes.append(np.polyfit(window['Year'], window['DOY_Retreat'], 1)[0])
    advance_slopes.append(np.polyfit(window['Year'], window['DOY_Advance'], 1)[0])

slope_df = pd.DataFrame({
    'End_Year': years,
    'Retreat_Slope': retreat_slopes,
    'Advance_Slope': advance_slopes
})

# === Plot Retreat ===
retreat_std = slope_df["Retreat_Slope"].std()
plt.figure(figsize=(10, 4))
plt.plot(slope_df["End_Year"], slope_df["Retreat_Slope"], color="black")
plt.fill_between(slope_df["End_Year"], -retreat_std, retreat_std, color="gray", alpha=0.3)
plt.scatter(slope_df["End_Year"], slope_df["Retreat_Slope"],
            color=["red" if v > 0 else "blue" for v in slope_df["Retreat_Slope"]])
plt.title("6-Year Trend in Retreat Timing")
plt.ylabel("days/year")
plt.xlabel("End year of 6-year window")
plt.axhline(0, linestyle="--", color="black")
plt.tight_layout()
plt.savefig("retreat_6yr_slopes.png", dpi=300)
plt.show()

# === Plot Advance ===
advance_std = slope_df["Advance_Slope"].std()
plt.figure(figsize=(10, 4))
plt.plot(slope_df["End_Year"], slope_df["Advance_Slope"], color="black")
plt.fill_between(slope_df["End_Year"], -advance_std, advance_std, color="gray", alpha=0.3)
plt.scatter(slope_df["End_Year"], slope_df["Advance_Slope"],
            color=["red" if v > 0 else "blue" for v in slope_df["Advance_Slope"]])
plt.title("6-Year Trend in Advance Timing")
plt.ylabel("days/year")
plt.xlabel("End year of 6-year window")
plt.axhline(0, linestyle="--", color="black")
plt.tight_layout()
plt.savefig("advance_6yr_slopes.png", dpi=300)
plt.show()