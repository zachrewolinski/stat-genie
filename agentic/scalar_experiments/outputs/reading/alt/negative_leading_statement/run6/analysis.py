import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# Define dyslexia subset using dyslexia_bin if available else dyslexia>0
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    dys_df = df[df['dyslexia'] > 0].copy()

# Keep rows with speed and reader_view
subset = dys_df[['speed', 'reader_view', 'num_words', 'Flesch_Kincaid', 'device', 'age', 'education', 'language', 'page_id', 'adjusted_running_time', 'running_time', 'scrolling_time']].copy()
subset = subset.replace([np.inf, -np.inf], np.nan)
subset = subset.dropna(subset=['speed', 'reader_view'])

# Basic descriptive stats
summary = subset.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).reset_index()

# Welch t-test
rv0 = subset[subset['reader_view'] == 0]['speed']
rv1 = subset[subset['reader_view'] == 1]['speed']

# guard against small sample size
if len(rv0) > 1 and len(rv1) > 1:
    ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    # Mann-Whitney U (two-sided)
    try:
        mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except Exception:
        mwu = None
else:
    ttest = None
    mwu = None

# Effect size (Hedges g)

def hedges_g(x, y):
    x = x.dropna()
    y = y.dropna()
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    # pooled sd
    s_pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if s_pooled == 0:
        return np.nan
    g = (x.mean() - y.mean()) / s_pooled
    # correction for small sample bias
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return g * correction

hg = hedges_g(rv1, rv0)

# Regression on log(speed) with controls to adjust for skew
subset = subset.copy()
subset = subset[subset['speed'] > 0]
subset['log_speed'] = np.log(subset['speed'])

# Build a formula; include categorical controls if enough data
# Use C() for categoricals
formula = 'log_speed ~ reader_view'
# add controls if columns present
for col in ['num_words', 'Flesch_Kincaid', 'age']:
    if col in subset.columns:
        formula += f' + {col}'
for col in ['device', 'education', 'language', 'page_id']:
    if col in subset.columns:
        formula += f' + C({col})'

model = smf.ols(formula, data=subset).fit(cov_type='HC3')

results = {
    'n_total_dyslexia': int(len(subset)),
    'n_reader_view_0': int(len(rv0)),
    'n_reader_view_1': int(len(rv1)),
    'summary_by_reader_view': summary.to_dict(orient='records'),
    'ttest': None if ttest is None else {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitney': None if mwu is None else {'statistic': float(mwu.statistic), 'pvalue': float(mwu.pvalue)},
    'hedges_g': None if np.isnan(hg) else float(hg),
    'regression': {
        'formula': formula,
        'coef_reader_view': float(model.params.get('reader_view', np.nan)),
        'pvalue_reader_view': float(model.pvalues.get('reader_view', np.nan)),
        'ci_reader_view': [float(x) for x in model.conf_int().loc['reader_view']] if 'reader_view' in model.params else None,
        'r2': float(model.rsquared),
        'n_obs': int(model.nobs)
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
