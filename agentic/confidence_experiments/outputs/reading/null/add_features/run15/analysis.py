import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Basic checks
print('rows', len(_df))

# Identify dyslexia group: dyslexia_bin==1 OR dyslexia in {1,2}
# We'll create a flag using available columns.

if 'dyslexia_bin' in _df.columns:
    dys_bin = _df['dyslexia_bin']
else:
    dys_bin = None

if 'dyslexia' in _df.columns:
    dys = _df['dyslexia']
else:
    dys = None

# Create flags
flag_bin = None
if dys_bin is not None:
    flag_bin = dys_bin == 1
flag_sev = None
if dys is not None:
    flag_sev = dys.isin([1,2])

if flag_bin is not None and flag_sev is not None:
    flag = flag_bin | flag_sev
elif flag_bin is not None:
    flag = flag_bin
elif flag_sev is not None:
    flag = flag_sev
else:
    raise ValueError('No dyslexia indicator found')

# Use speed variable
speed = _df['speed'] if 'speed' in _df.columns else None

# filter to dyslexia individuals
sub = _df[flag].copy()

# groups by reader_view
sub = sub[(sub['reader_view'].isin([0,1])) & sub['speed'].notna()]

print('dyslexia rows', len(sub))
print(sub['reader_view'].value_counts(dropna=False))

# summary stats
summary = sub.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
print(summary)

# t-test (Welch)
rv1 = sub[sub['reader_view']==1]['speed']
rv0 = sub[sub['reader_view']==0]['speed']

# log transform to reduce skew
rv1_log = np.log1p(rv1)
rv0_log = np.log1p(rv0)

print('skew raw', stats.skew(sub['speed']))

# Welch t-test raw
wt_raw = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('Welch t-test raw', wt_raw)

# Welch t-test log
wt_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')
print('Welch t-test log', wt_log)

# Mann-Whitney U
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('Mann-Whitney U', mw)
except Exception as e:
    print('Mann-Whitney error', e)

# Effect size (Cohen's d) on log
n1 = len(rv1_log)
n0 = len(rv0_log)

mean1 = rv1_log.mean()
mean0 = rv0_log.mean()
var1 = rv1_log.var(ddof=1)
var0 = rv0_log.var(ddof=1)

# pooled sd
sp = np.sqrt(((n1-1)*var1 + (n0-1)*var0)/(n1+n0-2))
cohen_d = (mean1-mean0)/sp if sp>0 else np.nan

print('cohen_d_log', cohen_d)

# Regression: speed ~ reader_view + num_words + device + age + gender + education + language + page_id
# We'll do log1p(speed) to mitigate skew, include basic controls, using statsmodels
import statsmodels.formula.api as smf

# prepare dataset with some controls
cols = ['speed','reader_view','num_words','device','age','gender','education','language','page_id','correct_rate','Flesch_Kincaid','retake_trial']

use_cols = [c for c in cols if c in sub.columns]
reg_df = sub[use_cols].copy()

# drop missing
reg_df = reg_df.dropna()

reg_df['log_speed'] = np.log1p(reg_df['speed'])

# build formula with categorical vars as C()
formula_parts = ['reader_view']
for c in ['num_words','age','gender','correct_rate','Flesch_Kincaid','retake_trial']:
    if c in reg_df.columns:
        formula_parts.append(c)
for c in ['device','education','language','page_id']:
    if c in reg_df.columns:
        formula_parts.append(f'C({c})')

formula = 'log_speed ~ ' + ' + '.join(formula_parts)

model = smf.ols(formula, data=reg_df).fit(cov_type='HC3')

print(model.summary().tables[1])

# extract reader_view coefficient and p-value
coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

print('coef reader_view', coef, 'p', pval, 'n', len(reg_df))

