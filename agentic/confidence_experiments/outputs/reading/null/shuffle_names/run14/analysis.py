import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Map columns based on info.json shuffle
participant_id = 'speed'          # UUIDs
reader_view = 'language'          # 0/1
reading_speed = 'running_time'    # numeric
# Dyslexia status: 0 none, 1 dyslexia, 2 severe
# Column "device" matches this distribution (1962 none, 264 dys, 174 severe)
dyslexia_status = 'device'

# Filter to dyslexic participants (1 or 2) and non-missing key fields
subset = df[df[dyslexia_status].isin([1.0, 2.0])].copy()
subset = subset[[participant_id, reader_view, reading_speed]].dropna()

# Ensure reader_view is binary 0/1
subset = subset[subset[reader_view].isin([0,1])]

# Summary stats
summary = subset.groupby(reader_view)[reading_speed].agg(['count','mean','median','std']).rename(index={0:'ReaderView=0',1:'ReaderView=1'})

# Mann-Whitney U test (two-sided)
rv0 = subset.loc[subset[reader_view]==0, reading_speed]
rv1 = subset.loc[subset[reader_view]==1, reading_speed]

u_stat, p_mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')

# Effect size: Cliff's delta
# compute using ranks
n1, n0 = len(rv1), len(rv0)
# use pairwise comparisons in an efficient way
# Cliff's delta = (number of greater - number of lesser) / (n1*n0)
# Use rank-based formula
all_vals = np.concatenate([rv1.values, rv0.values])
labels = np.array([1]*n1 + [0]*n0)
ranked = stats.rankdata(all_vals)
rank_sum1 = ranked[labels==1].sum()
# Mann-Whitney U for rv1 vs rv0
u1 = rank_sum1 - n1*(n1+1)/2
# delta
cliffs_delta = (2*u1)/(n1*n0) - 1

# Mixed effects model with random intercept for participant
# Use log-transform to reduce skew
subset['log_speed'] = np.log(subset[reading_speed])
try:
    model = smf.mixedlm(f"log_speed ~ {reader_view}", subset, groups=subset[participant_id])
    result = model.fit(reml=False, method='lbfgs', maxiter=200)
    coef = result.params.get(reader_view, np.nan)
    p_mixed = result.pvalues.get(reader_view, np.nan)
except Exception:
    coef = np.nan
    p_mixed = np.nan

# Save results to a json-like dict printed for manual use
print('Summary by reader_view (dyslexic only):')
print(summary)
print('\nMann-Whitney p:', p_mw, 'Cliff\'s delta:', cliffs_delta)
print('\nMixedLM log_speed coef (reader_view):', coef, 'p:', p_mixed)
print('Counts rv1, rv0:', n1, n0)
