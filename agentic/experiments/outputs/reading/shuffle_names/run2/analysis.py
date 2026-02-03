import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)

def bootstrap_diff(x, y, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    diffs = np.empty(iters)
    for i in range(iters):
        xs = rng.choice(x, size=len(x), replace=True)
        ys = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xs.mean() - ys.mean()
    return np.percentile(diffs, [2.5, 97.5])

# Load data
path = 'reading.csv'
raw = pd.read_csv(path)

# Map true semantics (column names are shuffled)
# reader_view indicator is 'language' (0/1)
# dyslexia status is 'device' (0=no,1=dyslexia,2=severe)
# reading speed is 'running_time'
# num_words likely 'num_words', readability likely 'uuid'

df = raw.copy()

# Basic cleaning
for col in ['language', 'device', 'running_time', 'num_words', 'uuid']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Focus on dyslexia or severe dyslexia
mask = df['device'].isin([1.0, 2.0]) & df['language'].isin([0, 1])
sub = df.loc[mask].copy()

# Remove non-positive speeds
sub = sub[sub['running_time'] > 0]

rv_on = sub[sub['language'] == 1]['running_time']
rv_off = sub[sub['language'] == 0]['running_time']

results = {}
results['n_on'] = len(rv_on)
results['n_off'] = len(rv_off)
results['mean_on'] = rv_on.mean()
results['mean_off'] = rv_off.mean()
results['median_on'] = rv_on.median()
results['median_off'] = rv_off.median()
results['diff_mean'] = results['mean_on'] - results['mean_off']
results['pct_change_mean'] = results['diff_mean'] / results['mean_off'] * 100
results['diff_median'] = results['median_on'] - results['median_off']

# Welch t-test
if len(rv_on) > 1 and len(rv_off) > 1:
    t_stat, p_val, _ = ttest_ind(rv_on, rv_off, usevar='unequal')
else:
    t_stat, p_val = np.nan, np.nan
results['t_stat'] = t_stat
results['p_val'] = p_val

# Effect size and bootstrap CI
if len(rv_on) > 2 and len(rv_off) > 2:
    results['cohen_d'] = cohen_d(rv_on, rv_off)
    ci_low, ci_high = bootstrap_diff(rv_on, rv_off)
    results['ci_low'] = ci_low
    results['ci_high'] = ci_high
else:
    results['cohen_d'] = np.nan
    results['ci_low'] = np.nan
    results['ci_high'] = np.nan

# Simple regression on log speed controlling for num_words and readability
sub_reg = sub.dropna(subset=['running_time', 'language', 'num_words', 'uuid']).copy()
sub_reg['log_speed'] = np.log(sub_reg['running_time'])
X = sub_reg[['language', 'num_words', 'uuid']]
X = sm.add_constant(X)
model = sm.OLS(sub_reg['log_speed'], X).fit()
results['reg_coef_reader_view'] = model.params.get('language', np.nan)
results['reg_p_reader_view'] = model.pvalues.get('language', np.nan)

# Save results for inspection
print('Dyslexia-only sample size:', len(sub))
print('Reader view ON n:', results['n_on'], 'OFF n:', results['n_off'])
print('Mean speed ON:', results['mean_on'])
print('Mean speed OFF:', results['mean_off'])
print('Mean diff (ON - OFF):', results['diff_mean'])
print('Median diff (ON - OFF):', results['diff_median'])
print('Pct change mean:', results['pct_change_mean'])
print('Welch t-test p-value:', results['p_val'])
print('Cohen d:', results['cohen_d'])
print('Bootstrap 95% CI (mean diff):', results['ci_low'], results['ci_high'])
print('Regression coef (log speed) reader_view:', results['reg_coef_reader_view'])
print('Regression p-value reader_view:', results['reg_p_reader_view'])
