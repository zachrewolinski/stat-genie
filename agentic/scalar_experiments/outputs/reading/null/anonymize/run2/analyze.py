import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Basic info
print('rows', len(_df))
print('columns', _df.columns.tolist())

# Identify potential reading speed measure: inspect feature20 and derived speed
# reading speed based on words per minute using feature7 (words) and feature5 (reading time minus scroll)
# feature5 is in ms. So wpm = words / (ms/60000)
_df['derived_wpm'] = _df['feature7'] / (_df['feature5'] / 60000.0)
_df['derived_wpm_total'] = _df['feature7'] / (_df['feature4'] / 60000.0)
_df['ms_per_word_reading'] = _df['feature5'] / _df['feature7']
_df['ms_per_word_total'] = _df['feature4'] / _df['feature7']

print('feature20 summary')
print(_df['feature20'].describe())
print('derived_wpm summary')
print(_df['derived_wpm'].describe())

# Correlation between feature20 and derived wpm
corr = _df[['feature20', 'derived_wpm']].corr().iloc[0,1]
print('corr feature20 vs derived_wpm', corr)
print('corr feature20 vs derived_wpm_total', _df[['feature20', 'derived_wpm_total']].corr().iloc[0,1])
print('corr feature20 vs ms_per_word_reading', _df[['feature20', 'ms_per_word_reading']].corr().iloc[0,1])
print('corr feature20 vs ms_per_word_total', _df[['feature20', 'ms_per_word_total']].corr().iloc[0,1])

# Focus on dyslexic individuals
# feature17 indicates dyslexia (1 yes, 0 no)
# feature3 indicates reader view active (1 yes, 0 no)

# Clean for non-missing and reasonable times
subset = _df.copy()
subset = subset.replace([np.inf, -np.inf], np.nan)

# We'll use feature20 as reading speed if it closely matches derived wpm
# but we still compute analysis for both to compare.

for speed_col in ['feature20', 'derived_wpm']:
    df = subset.loc[:, ['feature17', 'feature3', speed_col]].dropna()
    df = df[df['feature17'] == 1]  # dyslexic
    rv = df[df['feature3'] == 1][speed_col]
    no_rv = df[df['feature3'] == 0][speed_col]
    print('\n', speed_col, 'dyslexic counts', len(rv), len(no_rv))
    print('means', rv.mean(), no_rv.mean())
    # t-test (Welch)
    if len(rv) > 1 and len(no_rv) > 1:
        t_res = stats.ttest_ind(rv, no_rv, equal_var=False, nan_policy='omit')
        print('t-test', t_res)
    # Mann-Whitney
    if len(rv) > 0 and len(no_rv) > 0:
        mw = stats.mannwhitneyu(rv, no_rv, alternative='two-sided')
        print('mannwhitney', mw)

# Effect size for feature20
speed_col = 'feature20'
df = subset.loc[:, ['feature17', 'feature3', speed_col]].dropna()
df = df[df['feature17'] == 1]
rv = df[df['feature3'] == 1][speed_col]
no_rv = df[df['feature3'] == 0][speed_col]

# Cohen's d (Welch)
if len(rv) > 1 and len(no_rv) > 1:
    mean_diff = rv.mean() - no_rv.mean()
    s1 = rv.var(ddof=1)
    s2 = no_rv.var(ddof=1)
    n1 = len(rv)
    n2 = len(no_rv)
    # pooled sd for unequal sizes
    pooled_sd = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
    print('cohens_d', d)
