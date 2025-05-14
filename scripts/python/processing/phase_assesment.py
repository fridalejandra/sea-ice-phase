import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load your summary again
file_path = "/Users/fridaperez/Desktop/S_seaice_extent_daily_v3.0.csv"
df = pd.read_csv(file_path, skiprows=[1])
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df['Extent'] = pd.to_numeric(df['Extent'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['DOY'] = df['Date'].dt.dayofyear

def detect_onsets_per_year(df_year):
    df_year = df_year.sort_values('Date')
    df_year['dExtent'] = df_year['Extent'].diff()

    date_max = df_year.loc[df_year['Extent'].idxmax(), 'Date']
    date_min = df_year.loc[df_year['Extent'].idxmin(), 'Date']

    retreat = df_year[(df_year['Date'] > date_max) & (df_year['dExtent'] < 0)]
    advance = df_year[(df_year['Date'] > date_min) & (df_year['dExtent'] > 0)]

    if not retreat.empty and not advance.empty:
        date_retreat = retreat.iloc[0]['Date']
        date_advance = advance.iloc[0]['Date']
        return pd.Series({
            'DOY_Advance': date_advance.timetuple().tm_yday,
            'DOY_Retreat': date_retreat.timetuple().tm_yday
        })
    else:
        return pd.Series({'DOY_Advance': np.nan, 'DOY_Retreat': np.nan})

summary = df.groupby('Year').apply(detect_onsets_per_year).reset_index()
summary = summary[summary['Year'] > 1978]

# Calculate long-term means
mean_advance = summary['DOY_Advance'].mean(skipna=True)
mean_retreat = summary['DOY_Retreat'].mean(skipna=True)

def classify_phase(doy, mean_doy):
    if pd.isna(doy):
        return np.nan
    elif doy < (mean_doy - 5):
        return "Ahead of Phase"
    elif doy > (mean_doy + 5):
        return "Behind Phase"
    else:
        return "On Phase"

summary['Advance_Phase_Timing'] = summary['DOY_Advance'].apply(lambda x: classify_phase(x, mean_advance))
summary['Retreat_Phase_Timing'] = summary['DOY_Retreat'].apply(lambda x: classify_phase(x, mean_retreat))

# ---- Plotting with Annotations ----
colors = {'Ahead of Phase': 'steelblue', 'Behind Phase': 'tomato', 'On Phase': 'gray'}
# Advance plot with shading
plt.figure(figsize=(10, 5))
for label, color in colors.items():
    subset = summary[summary['Advance_Phase_Timing'] == label]
    plt.scatter(subset['Year'], subset['DOY_Advance'], label=label, color=color)
    for idx, row in subset.iterrows():
        if abs(row['DOY_Advance'] - mean_advance) > 10:
            plt.text(row['Year'], row['DOY_Advance'] + 2, str(row['Year']), fontsize=8, ha='center')

# Add shaded zone: ±5 days around mean
plt.axhline(mean_advance, linestyle='dashed', color='black')
plt.fill_between(summary['Year'], mean_advance - 5, mean_advance + 5, color='lightgray', alpha=0.4, label='±5 Day Window')

plt.title('Day of Year of Advance - Circumpolar (Annotated)')
plt.xlabel('Year')
plt.ylabel('DOY of Advance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/fridaperez/Desktop/Sea_Ice_Sheets/advance_phase_annotated.png", dpi=300)
plt.show()

# Retreat plot with shading
plt.figure(figsize=(10, 5))
for label, color in colors.items():
    subset = summary[summary['Retreat_Phase_Timing'] == label]
    plt.scatter(subset['Year'], subset['DOY_Retreat'], label=label, color=color)
    for idx, row in subset.iterrows():
        if abs(row['DOY_Retreat'] - mean_retreat) > 10:
            plt.text(row['Year'], row['DOY_Retreat'] + 2, str(row['Year']), fontsize=8, ha='center')

plt.axhline(mean_retreat, linestyle='dashed', color='black')
plt.fill_between(summary['Year'], mean_retreat - 5, mean_retreat + 5, color='lightgray', alpha=0.4, label='±5 Day Window')

plt.title('Day of Year of Retreat - Circumpolar (Annotated)')
plt.xlabel('Year')
plt.ylabel('DOY of Retreat')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/fridaperez/Desktop/Sea_Ice_Sheets/retreat_phase_annotated.png", dpi=300)
plt.show()
