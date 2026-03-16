import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'

df = pd.read_csv(path)

# Basic info
print('rows', df.shape[0], 'cols', df.shape[1])

# Focus on relevant columns
cols = ['speed','reader_view','dyslexia','dyslexia_bin','adjusted_running_time','running_time','num_words']
print('missing counts:')
print(df[cols].isna().sum())

# Determine dyslexia indicator
# Use dyslexia_bin if available else dyslexia>0
if 'dyslexia_bin' in df.columns:
    dyslexia_flag = df['dyslexia_bin']
else:
    dyslexia_flag = (df['dyslexia']>0).astype(int)

# Ensure numeric
for c in ['speed','reader_view']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Subset dyslexia individuals
sub = df[dyslexia_flag==1].copy()
print('dyslexia rows', sub.shape[0])

# Drop missing speed/reader_view
sub = sub.dropna(subset=['speed','reader_view'])

# Quick group stats
print(sub.groupby('reader_view')['speed'].agg(['count','mean','median','std']))

# Check distribution (skew) for speed
print('speed skew', sub['speed'].skew())

# Use log speed to mitigate skew
sub['log_speed'] = np.log(sub['speed'].clip(lower=1e-6))

print(sub.groupby('reader_view')['log_speed'].agg(['count','mean','median','std']))

# t-test on log speed
rv0 = sub[sub['reader_view']==0]['log_speed']
rv1 = sub[sub['reader_view']==1]['log_speed']

# Welch t-test
if len(rv0) > 1 and len(rv1) > 1:
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    print('welch t-test log speed: t=', tstat, 'p=', pval)

# Effect size (Cohen's d) on log speed
# compute pooled sd (for Welch using sample sizes?) use standard d with pooled variance
n0, n1 = len(rv0), len(rv1)
if n0 > 1 and n1 > 1:
    s0, s1 = rv0.std(ddof=1), rv1.std(ddof=1)
    pooled = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2)/(n0+n1-2))
    d = (rv1.mean() - rv0.mean())/pooled
    print('cohens d log speed', d)

# Regression controlling for page factors: num_words, Flesch_Kincaid, device, language, age
# Only include columns available
covars = []
for c in ['num_words','Flesch_Kincaid','age']:
    if c in df.columns:
        covars.append(c)

# Use categorical for page_id, device, language if present
cat_covars = [c for c in ['page_id','device','language'] if c in df.columns]

# Build formula
formula_parts = ['log_speed ~ reader_view']
if covars:
    formula_parts.append(' + ' + ' + '.join(covars))
if cat_covars:
    formula_parts.append(' + ' + ' + '.join([f'C({c})' for c in cat_covars]))

formula = ''.join(formula_parts)

print('formula', formula)

# Fit OLS on dyslexia subset
# drop missing covars
model_df = sub.copy()
model_df = model_df.dropna(subset=['log_speed','reader_view'] + covars + cat_covars)

if model_df.shape[0] > 20:
    try:
        model = smf.ols(formula, data=model_df).fit(cov_type='HC3')
        print(model.summary().tables[1])
    except Exception as e:
        print('regression failed', e)

# Also compute non-parametric test (Mann-Whitney)
if n0 > 1 and n1 > 1:
    u, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('mannwhitney log speed: U=', u, 'p=', p_u)

# proportion speed increase (median ratio)
if n0>0 and n1>0:
    ratio = np.exp(rv1.median() - rv0.median())
    print('median speed ratio (rv1/rv0) using log', ratio)

