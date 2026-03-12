import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.formula.api import ols

# Load data

df = pd.read_csv('reading.csv')

# Focus on dyslexic participants
# dyslexia_bin: 1 indicates dyslexia
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] >= 1].copy()

# Basic groups
rv0 = dys[dys['reader_view'] == 0]['speed']
rv1 = dys[dys['reader_view'] == 1]['speed']

# Summary stats
summary = {
    'n_dys': len(dys),
    'n_rv0': rv0.shape[0],
    'n_rv1': rv1.shape[0],
    'mean_rv0': rv0.mean(),
    'mean_rv1': rv1.mean(),
    'median_rv0': rv0.median(),
    'median_rv1': rv1.median(),
    'std_rv0': rv0.std(),
    'std_rv1': rv1.std(),
}

# t-test (Welch)

stat, pval, dfree = ttest_ind(rv1, rv0, usevar='unequal')

# Effect size (Cohen's d)

# Pooled SD using Welch's formula? We'll compute standard Cohen's d with pooled SD
n1, n0 = rv1.shape[0], rv0.shape[0]
if n1 > 1 and n0 > 1:
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/(n1+n0-2))
    d = (rv1.mean() - rv0.mean())/sp if sp != 0 else np.nan
else:
    d = np.nan

# Regression with controls and clustered SEs by uuid

# Clean missing values for selected columns
cols = ['speed', 'reader_view', 'num_words', 'page_id', 'device', 'language', 'age', 'gender', 'education', 'english_native', 'uuid']
cols = [c for c in cols if c in dys.columns]
reg_df = dys[cols].dropna().copy()

# Build formula
# C() for categorical
formula_parts = ['reader_view']
for c in ['num_words', 'age']:
    if c in reg_df.columns:
        formula_parts.append(c)
# categorical controls
for c in ['page_id', 'device', 'language', 'gender', 'education', 'english_native']:
    if c in reg_df.columns:
        formula_parts.append(f'C({c})')
formula = 'speed ~ ' + ' + '.join(formula_parts)

if len(reg_df) > 0:
    model = ols(formula, data=reg_df).fit(cov_type='cluster', cov_kwds={'groups': reg_df['uuid']})
    coef = model.params.get('reader_view', np.nan)
    pval_reg = model.pvalues.get('reader_view', np.nan)
    ci_low, ci_high = model.conf_int().loc['reader_view'] if 'reader_view' in model.params else (np.nan, np.nan)
    n_reg = model.nobs
else:
    coef = pval_reg = ci_low = ci_high = np.nan
    n_reg = 0

# Print results
print('SUMMARY')
for k, v in summary.items():
    print(f'{k}: {v}')
print('t-test')
print('t_stat:', stat, 'pval:', pval, 'df:', dfree)
print("cohen_d:", d)
print('regression')
print('n_reg:', n_reg)
print('coef_reader_view:', coef)
print('pval_reader_view:', pval_reg)
print('ci_reader_view:', (ci_low, ci_high))
