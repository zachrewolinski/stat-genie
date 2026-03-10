import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Basic column check
print('Columns:', df.columns.tolist())

# Identify key columns based on observed structure
participant_col = 'speed'
reader_view_col = 'language'  # 0/1 indicator
reading_speed_col = 'running_time'  # computed wpm

# Dyslexia status column candidates
# device has values 0/1/2 (likely dyslexia status), dyslexia has 0/1/2 (likely gender)
print('\nValue counts for device:', df['device'].value_counts(dropna=False).head())
print('Value counts for dyslexia:', df['dyslexia'].value_counts(dropna=False).head())
print('Value counts for dyslexia_bin:', df['dyslexia_bin'].value_counts(dropna=False).head())

# Check alignment between device and dyslexia_bin
ct = pd.crosstab(df['device'], df['dyslexia_bin'])
print('\nCrosstab device vs dyslexia_bin:\n', ct)

# Define dyslexia indicator
# Use dyslexia_bin if it matches device>0; else use device>0
if set(df['dyslexia_bin'].dropna().unique()) <= {0.0, 1.0}:
    # See if dyslexia_bin matches device>0 for most rows
    device_bin = (df['device'] > 0).astype(float)
    match_rate = (device_bin == df['dyslexia_bin']).mean()
    print('\nMatch rate between device>0 and dyslexia_bin:', match_rate)
    if match_rate > 0.95:
        dyslexia_bin = df['dyslexia_bin']
        dyslexia_source = 'dyslexia_bin'
    else:
        dyslexia_bin = (df['device'] > 0).astype(int)
        dyslexia_source = 'device>0'
else:
    dyslexia_bin = (df['device'] > 0).astype(int)
    dyslexia_source = 'device>0'

print('Using dyslexia indicator from:', dyslexia_source)

# Filter to dyslexic individuals

df_dys = df[dyslexia_bin == 1].copy()

# Ensure reader_view is binary 0/1
print('Reader view unique:', sorted(df_dys[reader_view_col].dropna().unique())[:10])

# Summary stats by reader_view
summary = df_dys.groupby(reader_view_col)[reading_speed_col].agg(['count','mean','std'])
print('\nSummary by reader_view (dyslexic only):\n', summary)

# Participant-level means for paired comparison
pivot = df_dys.pivot_table(index=participant_col, columns=reader_view_col, values=reading_speed_col, aggfunc='mean')
# Keep only participants with both conditions
pivot = pivot.dropna()

print('\nParticipants with both conditions:', len(pivot))

if len(pivot) > 1:
    diff = pivot[1] - pivot[0]
    # Paired t-test (one-sample t-test on differences)
    tstat, pval = stats.ttest_1samp(diff, 0.0, nan_policy='omit')
    # Effect size for paired samples (Cohen's d)
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
    print('\nPaired diff mean (reader_view=1 minus 0):', diff.mean())
    print('Paired t-test: t=', tstat, 'p=', pval)
    print('Paired Cohen d:', d)
else:
    print('Not enough paired participants for paired test.')

# Also run mixed effects model as robustness (if possible)
try:
    import statsmodels.formula.api as smf
    # Use only dyslexic rows, reader_view as numeric
    # Add random intercept for participant
    md = smf.mixedlm(f"{reading_speed_col} ~ {reader_view_col}", df_dys, groups=df_dys[participant_col])
    mdf = md.fit(reml=False)
    print('\nMixedLM result:\n', mdf.summary())
except Exception as e:
    print('\nMixedLM failed:', e)
