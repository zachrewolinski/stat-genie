import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df), 'cols', df.shape[1])

# Derive reading speed from words and reading time (excluding scrolling)
# feature7: number of words
# feature5: reading time excluding scrolling in ms
# Avoid division by zero

df['derived_speed_wpm'] = df['feature7'] / (df['feature5'] / 60000.0)

# Compare derived speed with feature20
# Clean possible inf or NaN
valid = df[['derived_speed_wpm', 'feature20']].replace([np.inf, -np.inf], np.nan).dropna()
if len(valid) > 0:
    corr = valid.corr().iloc[0,1]
    print('corr(derived_speed_wpm, feature20)=', corr)
    print('feature20 summary', df['feature20'].describe())
    print('derived speed summary', df['derived_speed_wpm'].describe())

# Dyslexia subset
# feature12: 0 no dyslexia, 1 dyslexia, 2 severe dyslexia
# feature17: 1 dyslexia yes, 0 no

# We'll use feature12 >=1 as dyslexia (including severe)

subset = df[df['feature12'] >= 1].copy()
print('dyslexia subset n', len(subset))

# Compare reading speed between reader view on/off
# feature3: reader view activated (1) or not (0)

# Use feature20 as provided reading speed and derived as check

for speed_col in ['feature20', 'derived_speed_wpm']:
    data = subset[[speed_col, 'feature3']].replace([np.inf, -np.inf], np.nan).dropna()
    group0 = data[data['feature3'] == 0][speed_col]
    group1 = data[data['feature3'] == 1][speed_col]
    print('\nSpeed col', speed_col)
    print('n0', len(group0), 'n1', len(group1))
    print('mean0', group0.mean(), 'mean1', group1.mean())
    print('median0', group0.median(), 'median1', group1.median())
    # Welch t-test
    if len(group0) > 1 and len(group1) > 1:
        tstat, pval = stats.ttest_ind(group1, group0, equal_var=False, nan_policy='omit')
        print('welch t-test t', tstat, 'p', pval)
        # effect size: Cohen's d (using pooled SD for approximated interpretation)
        s1 = group1.std(ddof=1)
        s0 = group0.std(ddof=1)
        n1 = len(group1)
        n0 = len(group0)
        # Pooled SD
        sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2)) if (n1+n0-2)>0 else np.nan
        d = (group1.mean() - group0.mean()) / sp if sp > 0 else np.nan
        print('cohen_d', d)
    # Non-parametric test
    if len(group0) > 1 and len(group1) > 1:
        u_stat, p_u = stats.mannwhitneyu(group1, group0, alternative='two-sided')
        print('mannwhitney_u', u_stat, 'p', p_u)

# Also check within dyslexia status using feature17
subset2 = df[df['feature17'] == 1].copy()
print('\nfeature17 dyslexia subset n', len(subset2))

for speed_col in ['feature20', 'derived_speed_wpm']:
    data = subset2[[speed_col, 'feature3']].replace([np.inf, -np.inf], np.nan).dropna()
    group0 = data[data['feature3'] == 0][speed_col]
    group1 = data[data['feature3'] == 1][speed_col]
    print('\nSpeed col', speed_col, '(feature17 subset)')
    print('n0', len(group0), 'n1', len(group1))
    print('mean0', group0.mean(), 'mean1', group1.mean())
    if len(group0) > 1 and len(group1) > 1:
        tstat, pval = stats.ttest_ind(group1, group0, equal_var=False, nan_policy='omit')
        print('welch t-test t', tstat, 'p', pval)
        s1 = group1.std(ddof=1)
        s0 = group0.std(ddof=1)
        n1 = len(group1)
        n0 = len(group0)
        sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2)) if (n1+n0-2)>0 else np.nan
        d = (group1.mean() - group0.mean()) / sp if sp > 0 else np.nan
        print('cohen_d', d)
        u_stat, p_u = stats.mannwhitneyu(group1, group0, alternative='two-sided')
        print('mannwhitney_u', u_stat, 'p', p_u)

