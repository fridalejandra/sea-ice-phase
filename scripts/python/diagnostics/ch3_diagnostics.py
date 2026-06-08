import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr

# ── Load data ──────────────────────────────────────────────────────────────────
params = pd.read_csv('~/Research/repos/sea-ice-phase/scripts/R/Ch3/data/annual_params.csv')

# Compute a simple SIE anomaly proxy: mean of max+min extent, demeaned
params['sie_level'] = (params['max_extent'] + params['min_extent']) / 2
sector_means = params.groupby('sector')['sie_level'].mean()
params['sie_anomaly'] = params.apply(
    lambda r: r['sie_level'] - sector_means[r['sector']], axis=1
)

SECTORS = params['sector'].unique()
CUTOFF = 2016

sector_labels = {
    'SIE_Weddell': 'Weddell',
    'SIE_Amundsen_Bellingshausen': 'ABS',
    'SIE_Ross': 'Ross',
    'SIE_East_Antarctica': 'East Antarctica',
    'SIE_King_Haakon': 'King Haakon',
    'SIE_circumpolar': 'Circumpolar'
}

# ── Diagnostic 1: Phase & Amplitude vs SIE anomaly, + scatter ─────────────────
print("Running Diagnostic 1...")
fig, axes = plt.subplots(len(SECTORS), 3, figsize=(16, 3.5 * len(SECTORS)))
fig.suptitle('Diagnostic 1: Phase & Amplitude vs SIE Anomaly', fontsize=13, y=1.01)

for i, sec in enumerate(SECTORS):
    df = params[params['sector'] == sec].sort_values('Year')
    pre = df[df['Year'] <= CUTOFF]
    post = df[df['Year'] > CUTOFF]
    label = sector_labels.get(sec, sec)

    # Panel 1: Phase anomaly vs SIE anomaly
    ax = axes[i, 0]
    ax.plot(df['Year'], df['max_doy_raw_anom'], 'b-o', markersize=3, label='Phase anom (DOY)')
    ax2 = ax.twinx()
    ax2.plot(df['Year'], df['sie_anomaly'], 'r-o', markersize=3, alpha=0.6, label='SIE anomaly')
    ax.axvline(CUTOFF, color='k', linestyle='--', alpha=0.4)
    ax.set_title(f'{label}: Phase vs SIE anomaly', fontsize=9)
    ax.set_ylabel('Phase anomaly (days)', color='b', fontsize=8)
    ax2.set_ylabel('SIE anomaly', color='r', fontsize=8)
    ax.tick_params(axis='y', labelcolor='b', labelsize=7)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=7)

    # Panel 2: Amplitude anomaly vs SIE anomaly
    ax = axes[i, 1]
    ax.plot(df['Year'], df['amplitude_raw_anom'], 'g-o', markersize=3, label='Amp anom')
    ax2 = ax.twinx()
    ax2.plot(df['Year'], df['sie_anomaly'], 'r-o', markersize=3, alpha=0.6)
    ax.axvline(CUTOFF, color='k', linestyle='--', alpha=0.4)
    ax.set_title(f'{label}: Amplitude vs SIE anomaly', fontsize=9)
    ax.set_ylabel('Amplitude anomaly', color='g', fontsize=8)
    ax2.set_ylabel('SIE anomaly', color='r', fontsize=8)
    ax.tick_params(axis='y', labelcolor='g', labelsize=7)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=7)

    # Panel 3: Phase vs Amplitude scatter, pre/post 2016
    ax = axes[i, 2]
    ax.scatter(pre['max_doy_raw_anom'], pre['amplitude_raw_anom'],
               c='steelblue', label='≤2016', alpha=0.7, s=30)
    ax.scatter(post['max_doy_raw_anom'], post['amplitude_raw_anom'],
               c='firebrick', label='>2016', alpha=0.85, s=50, zorder=5)
    for _, row in df[df['Year'] >= 2014].iterrows():
        ax.annotate(str(int(row['Year'])),
                    (row['max_doy_raw_anom'], row['amplitude_raw_anom']),
                    fontsize=6, alpha=0.85,
                    xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Phase anomaly (days)', fontsize=8)
    ax.set_ylabel('Amplitude anomaly', fontsize=8)
    ax.set_title(f'{label}: Phase vs Amplitude', fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('diagnostic1_phase_amp_anomaly.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic1_phase_amp_anomaly.png")
plt.close()

# ── Diagnostic 2: Correlation table pre/post 2016 ─────────────────────────────
print("Running Diagnostic 2...")
results = []
for sec in SECTORS:
    df = params[params['sector'] == sec].sort_values('Year')
    pre = df[df['Year'] <= CUTOFF]
    post = df[df['Year'] > CUTOFF]

    for period, d in [('full', df), ('pre_2016', pre), ('post_2016', post)]:
        if len(d) < 5:
            continue
        r_phase, p_phase = spearmanr(d['max_doy_raw_anom'], d['sie_anomaly'])
        r_amp, p_amp = spearmanr(d['amplitude_raw_anom'], d['sie_anomaly'])
        r_pa, p_pa = spearmanr(d['max_doy_raw_anom'], d['amplitude_raw_anom'])
        results.append({
            'sector': sector_labels.get(sec, sec),
            'period': period,
            'r_phase_anom': round(r_phase, 2), 'p_phase': round(p_phase, 3),
            'r_amp_anom': round(r_amp, 2), 'p_amp': round(p_amp, 3),
            'r_phase_amp': round(r_pa, 2), 'p_pa': round(p_pa, 3),
            'n': len(d)
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv('diagnostic2_correlations.csv', index=False)
print("  Saved diagnostic2_correlations.csv")

# ── Diagnostic 3: Rolling correlations ────────────────────────────────────────
print("Running Diagnostic 3...")
WINDOW = 10

fig, axes = plt.subplots(len(SECTORS), 1, figsize=(11, 3 * len(SECTORS)))
fig.suptitle(f'Diagnostic 3: Rolling {WINDOW}-year Spearman r with SIE anomaly', fontsize=12)

for i, sec in enumerate(SECTORS):
    df = params[params['sector'] == sec].sort_values('Year').reset_index(drop=True)
    label = sector_labels.get(sec, sec)

    roll_r_phase, roll_r_amp, years = [], [], []
    for j in range(WINDOW, len(df) + 1):
        w = df.iloc[j - WINDOW:j]
        r_p, _ = spearmanr(w['max_doy_raw_anom'], w['sie_anomaly'])
        r_a, _ = spearmanr(w['amplitude_raw_anom'], w['sie_anomaly'])
        roll_r_phase.append(r_p)
        roll_r_amp.append(r_a)
        years.append(df.iloc[j - 1]['Year'])

    axes[i].plot(years, roll_r_phase, 'b-o', markersize=3, label='phase–SIE r')
    axes[i].plot(years, roll_r_amp, 'g-o', markersize=3, label='amp–SIE r')
    axes[i].axhline(0, color='k', linewidth=0.5, alpha=0.4)
    axes[i].axvline(CUTOFF, color='r', linestyle='--', alpha=0.5, label='2016')
    axes[i].fill_between(years, -0.4, 0.4, alpha=0.05, color='gray',
                         label='|r|<0.4')
    axes[i].set_title(label, fontsize=9)
    axes[i].set_ylabel('Spearman r', fontsize=8)
    axes[i].set_ylim(-1, 1)
    axes[i].legend(fontsize=7, loc='upper left')
    axes[i].tick_params(labelsize=7)

axes[-1].set_xlabel('Year (end of window)', fontsize=9)
plt.tight_layout()
plt.savefig('diagnostic3_rolling_correlations.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic3_rolling_correlations.png")
plt.close()

print("\nDiagnostics 1-3 complete.")

# ── Diagnostic 4 setup ────────────────────────────────────────────────────────
from scipy.stats import rankdata

POST = params[params['Year'] > CUTOFF].copy()

COLORS = {
    'SIE_Weddell': '#1f77b4',
    'SIE_Amundsen_Bellingshausen': '#d62728',
    'SIE_Ross': '#2ca02c',
    'SIE_East_Antarctica': '#ff7f0e',
    'SIE_King_Haakon': '#9467bd',
    'SIE_circumpolar': '#8c564b'
}
SECTORS_ORDERED = list(sector_labels.keys())

# ── Diagnostic 4a: Anomaly looks uniform, components look heterogeneous ────────
print("Running Diagnostic 4a...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Diagnostic 4a: One anomaly, multiple stories (post-2016)',
             fontsize=12)

for sec in SECTORS_ORDERED:
    df = POST[POST['sector'] == sec].sort_values('Year')
    label = sector_labels[sec]
    col = COLORS[sec]
    lw = 2.5 if sec == 'SIE_circumpolar' else 1.5
    ls = '--' if sec == 'SIE_circumpolar' else '-'

    axes[0].plot(df['Year'], df['sie_anomaly'],
                 color=col, linewidth=lw, linestyle=ls,
                 marker='o', markersize=4, label=label)
    axes[1].plot(df['Year'], df['max_doy_raw_anom'],
                 color=col, linewidth=lw, linestyle=ls,
                 marker='o', markersize=4, label=label)
    axes[2].plot(df['Year'], df['amplitude_raw_anom'],
                 color=col, linewidth=lw, linestyle=ls,
                 marker='o', markersize=4, label=label)

for ax, ylabel, title in zip(
        axes,
        ['SIE anomaly\n(10⁶ km²)', 'Phase anomaly\n(days)', 'Amplitude anomaly\n(10⁶ km²)'],
        ['SIE anomaly — sectors look similar (uniform decline)',
         'Phase anomaly — sectors diverge',
         'Amplitude anomaly — sectors diverge']
):
    ax.axhline(0, color='k', linewidth=0.7, alpha=0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc='left')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[2].set_xlabel('Year', fontsize=9)
plt.tight_layout()
plt.savefig('diagnostic4a_uniform_vs_heterogeneous.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic4a_uniform_vs_heterogeneous.png")
plt.close()

# ── Diagnostic 4b: Ranking comparison ─────────────────────────────────────────
print("Running Diagnostic 4b...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Diagnostic 4b: Do anomaly rankings agree with component rankings?\n'
             'Rank 1 = most negative. P≠ / A≠ = anomaly misleads.',
             fontsize=11)

for i, sec in enumerate(SECTORS_ORDERED):
    df = POST[POST['sector'] == sec].sort_values('Year').copy()
    label = sector_labels[sec]
    ax = axes.flatten()[i]

    df['rank_anom'] = rankdata(df['sie_anomaly'])
    df['rank_phase'] = rankdata(df['max_doy_raw_anom'])
    df['rank_amp'] = rankdata(df['amplitude_raw_anom'])

    years = df['Year'].values
    x = np.arange(len(years))
    width = 0.25

    ax.bar(x - width, df['rank_anom'], width, label='SIE anomaly rank',
           color='salmon', alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.bar(x, df['rank_phase'], width, label='Phase rank',
           color='steelblue', alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.bar(x + width, df['rank_amp'], width, label='Amplitude rank',
           color='mediumseagreen', alpha=0.85, edgecolor='k', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=7)
    ax.set_ylabel('Rank (1 = most negative)', fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for _, row in df.iterrows():
        idx = list(years).index(row['Year'])
        if abs(row['rank_anom'] - row['rank_phase']) >= 3:
            ax.annotate('P≠', (idx - width, row['rank_anom'] + 0.1),
                        fontsize=6, color='steelblue', ha='center')
        if abs(row['rank_anom'] - row['rank_amp']) >= 3:
            ax.annotate('A≠', (idx + width, row['rank_anom'] + 0.1),
                        fontsize=6, color='darkgreen', ha='center')

plt.tight_layout()
plt.savefig('diagnostic4b_ranking_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic4b_ranking_comparison.png")
plt.close()

# ── Diagnostic 4c: Phase-amplitude independence across time ───────────────────
print("Running Diagnostic 4c...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f'Diagnostic 4c: Are phase and amplitude independent?\n'
             f'Rolling {WINDOW}-year r (phase vs amplitude). '
             'Near zero = independent = decomposition adds information.',
             fontsize=10)

for i, sec in enumerate(SECTORS_ORDERED):
    df = params[params['sector'] == sec].sort_values('Year').reset_index(drop=True)
    label = sector_labels[sec]
    ax = axes.flatten()[i]

    roll_r, years = [], []
    for j in range(WINDOW, len(df) + 1):
        w = df.iloc[j - WINDOW:j]
        r, _ = spearmanr(w['max_doy_raw_anom'], w['amplitude_raw_anom'])
        roll_r.append(r)
        years.append(df.iloc[j - 1]['Year'])

    ax.plot(years, roll_r, color=COLORS[sec], linewidth=2, marker='o', markersize=3)
    ax.axhline(0, color='k', linewidth=0.7, alpha=0.4)
    ax.axhline(0.4, color='gray', linewidth=0.7, linestyle=':', alpha=0.6)
    ax.axhline(-0.4, color='gray', linewidth=0.7, linestyle=':', alpha=0.6)
    ax.axvline(CUTOFF, color='r', linewidth=1.2, linestyle='--', alpha=0.6, label='2016')
    ax.fill_between(years, -0.4, 0.4, alpha=0.07, color='gray')
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('r (phase vs amplitude)', fontsize=8)
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    pre_mean = np.mean([r for r, y in zip(roll_r, years) if y <= CUTOFF])
    post_mean = np.mean([r for r, y in zip(roll_r, years) if y > CUTOFF])
    ax.annotate(f'pre mean: {pre_mean:.2f}', xy=(0.03, 0.92),
                xycoords='axes fraction', fontsize=7, color='gray')
    ax.annotate(f'post mean: {post_mean:.2f}', xy=(0.03, 0.82),
                xycoords='axes fraction', fontsize=7, color='firebrick')

fig.supxlabel('Year (end of window)', fontsize=9)
plt.tight_layout()
plt.savefig('diagnostic4c_phase_amp_independence.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic4c_phase_amp_independence.png")
plt.close()

print("\nDiagnostics 1-4 complete.")

# ── Diagnostic 5a: Phase-amplitude scatter, SIE anomaly as color ──────────────
# Each point = one sector-year post-2016
# x = phase anomaly, y = amplitude anomaly, color = SIE anomaly magnitude
# Points near x-axis = phase-driven; near y-axis = amplitude-driven; diagonal = both
print("Running Diagnostic 5a...")

import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(10, 8))

# collect all post-2016 sector-year points
scatter_data = []
for sec in SECTORS_ORDERED:
    df = POST[POST['sector'] == sec].sort_values('Year')
    for _, row in df.iterrows():
        scatter_data.append({
            'sector': sector_labels[sec],
            'year': int(row['Year']),
            'phase_anom': row['max_doy_raw_anom'],
            'amp_anom': row['amplitude_raw_anom'],
            'sie_anom': row['sie_anomaly'],
            'color': COLORS[sec]
        })

sdf = pd.DataFrame(scatter_data)

# color by SIE anomaly magnitude using diverging colormap
vmax = max(abs(sdf['sie_anom'].max()), abs(sdf['sie_anom'].min()))
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu

sc = ax.scatter(sdf['phase_anom'], sdf['amp_anom'],
                c=sdf['sie_anom'], cmap=cmap, norm=norm,
                s=120, edgecolors='k', linewidths=0.5, zorder=3)

# label each point with year + sector initial
for _, row in sdf.iterrows():
    sec_init = row['sector'][0]  # first letter
    ax.annotate(f"{row['year']}\n{row['sector'][:3]}",
                (row['phase_anom'], row['amp_anom']),
                fontsize=5.5, ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points', alpha=0.8)

# reference lines and quadrant labels
ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
ax.axvline(0, color='k', linewidth=0.8, alpha=0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
pad = 0.04
ax.text(xlim[1] - pad * (xlim[1] - xlim[0]), pad * (ylim[1] - ylim[0]) + ylim[0],
        'Late peak\nHigh amplitude', fontsize=7, ha='right', color='gray', style='italic')
ax.text(xlim[0] + pad * (xlim[1] - xlim[0]), pad * (ylim[1] - ylim[0]) + ylim[0],
        'Early peak\nHigh amplitude', fontsize=7, ha='left', color='gray', style='italic')
ax.text(xlim[1] - pad * (xlim[1] - xlim[0]), ylim[1] - pad * (ylim[1] - ylim[0]),
        'Late peak\nLow amplitude', fontsize=7, ha='right', color='gray', style='italic')
ax.text(xlim[0] + pad * (xlim[1] - xlim[0]), ylim[1] - pad * (ylim[1] - ylim[0]),
        'Early peak\nLow amplitude', fontsize=7, ha='left', color='gray', style='italic')

cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
cbar.set_label('SIE anomaly (10⁶ km²)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xlabel('Phase anomaly (days, +ve = late peak)', fontsize=10)
ax.set_ylabel('Amplitude anomaly (10⁶ km², +ve = larger range)', fontsize=10)
ax.set_title('Diagnostic 5a: Phase–amplitude space, post-2016\n'
             'Color = SIE anomaly. Same color, different position = anomaly obscures mechanism.',
             fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('diagnostic5a_phase_amp_scatter.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic5a_phase_amp_scatter.png")
plt.close()

# ── Diagnostic 5b: Decomposition bar — how much does each component explain? ──
# For each post-2016 year × sector, show SIE anomaly as a bar,
# then overlay how much phase and amplitude account for it.
# Method: partial correlation / proportion of variance approach.
# Simple version: sign-weighted fraction using standardised anomalies.
print("Running Diagnostic 5b...")

from scipy.stats import zscore

# Standardise within each sector across full record for comparability
for sec in SECTORS_ORDERED:
    idx = params['sector'] == sec
    params.loc[idx, 'phase_z'] = zscore(params.loc[idx, 'max_doy_raw_anom'], nan_policy='omit')
    params.loc[idx, 'amp_z'] = zscore(params.loc[idx, 'amplitude_raw_anom'], nan_policy='omit')
    params.loc[idx, 'sie_z'] = zscore(params.loc[idx, 'sie_anomaly'], nan_policy='omit')

POST = params[params['Year'] > CUTOFF].copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Diagnostic 5b: Phase vs amplitude contribution to SIE anomaly (post-2016)\n'
             'Bars = standardised SIE anomaly. '
             'Overlaid lines = phase (blue) and amplitude (green) standardised anomalies.',
             fontsize=10)

for i, sec in enumerate(SECTORS_ORDERED):
    df = POST[POST['sector'] == sec].sort_values('Year').copy()
    label = sector_labels[sec]
    ax = axes.flatten()[i]

    years = df['Year'].values
    x = np.arange(len(years))

    # SIE anomaly bar
    bar_colors = ['salmon' if v < 0 else 'lightblue' for v in df['sie_z']]
    ax.bar(x, df['sie_z'], color=bar_colors, alpha=0.7,
           edgecolor='k', linewidth=0.5, label='SIE anomaly (z)', zorder=2)

    # Phase and amplitude overlaid as lines
    ax.plot(x, df['phase_z'], 'b-o', markersize=5, linewidth=1.5,
            label='Phase (z)', zorder=3)
    ax.plot(x, df['amp_z'], 'g-o', markersize=5, linewidth=1.5,
            label='Amplitude (z)', zorder=3)

    ax.axhline(0, color='k', linewidth=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=7)
    ax.set_ylabel('Standardised anomaly (z)', fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # Annotate years where phase and amplitude point in opposite directions
    for j, (_, row) in enumerate(df.iterrows()):
        if np.sign(row['phase_z']) != np.sign(row['amp_z']) and \
                abs(row['phase_z']) > 0.5 and abs(row['amp_z']) > 0.5:
            ax.annotate('↕', (j, max(abs(row['phase_z']), abs(row['amp_z'])) + 0.1),
                        fontsize=10, ha='center', color='purple',
                        annotation_clip=False)

plt.tight_layout()
plt.savefig('diagnostic5b_decomposition_bars.png', dpi=150, bbox_inches='tight')
print("  Saved diagnostic5b_decomposition_bars.png")
plt.close()

print("\nAll diagnostics complete.")
print("Key things to look for:")
print("  Diag 5a: points with similar SIE anomaly color but different positions in")
print("           phase-amplitude space = anomaly obscures the mechanism")
print("  Diag 5b: years where blue (phase) and green (amplitude) lines diverge in sign")
print("           = the anomaly bar is a mixture that cannot be interpreted without decomposition")
print("           ↕ symbol marks years where phase and amplitude actively oppose each other")
