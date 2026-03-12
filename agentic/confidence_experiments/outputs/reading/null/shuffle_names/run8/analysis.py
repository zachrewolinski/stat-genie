import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Column mapping inferred from metadata/value patterns
# reader_view indicator appears to be the binary 'language' column (0/1)
reader_view_col = 'language'
# dyslexia status appears to be in 'device' (0=no,1=dyslexia,2=severe)
dyslexia_status_col = 'device'
# reading speed appears to be in 'running_time' (values ~200-400 typical, likely WPM)
speed_col = 'running_time'

# Prepare analysis subset: individuals with dyslexia (device > 0)
subset = df[[reader_view_col, dyslexia_status_col, speed_col]].dropna()
subset = subset[subset[dyslexia_status_col] > 0]

# Split by reader view
rv1 = subset[subset[reader_view_col] == 1][speed_col]
rv0 = subset[subset[reader_view_col] == 0][speed_col]

# Basic stats
stats_summary = {
    'n_dyslexia_total': len(subset),
    'n_rv1': len(rv1),
    'n_rv0': len(rv0),
    'mean_rv1': rv1.mean(),
    'mean_rv0': rv0.mean(),
    'median_rv1': rv1.median(),
    'median_rv0': rv0.median(),
    'std_rv1': rv1.std(ddof=1),
    'std_rv0': rv0.std(ddof=1),
}

# Welch's t-test
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# Effect size: Cohen's d (using pooled SD with unequal n)
def cohens_d(x, y):
    x = x.dropna().to_numpy()
    y = y.dropna().to_numpy()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if s_pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / s_pooled

d_val = cohens_d(rv1, rv0)

# Robustness: simple regression within dyslexia subset
# speed ~ reader_view (binary)
reg_result = None
if subset[reader_view_col].nunique() > 1:
    model = smf.ols(f"{speed_col} ~ {reader_view_col}", data=subset).fit()
    reg_result = {
        'coef_reader_view': model.params.get(reader_view_col, np.nan),
        'p_value_reader_view': model.pvalues.get(reader_view_col, np.nan),
        'n': int(model.nobs),
        'r2': model.rsquared,
    }

print('SUMMARY')
print(stats_summary)
print('t_stat', t_stat, 'p_val', p_val, 'cohens_d', d_val)
print('REG', reg_result)

# Save key metrics for downstream write-up
pd.Series(stats_summary).to_csv('analysis_metrics.csv')
