import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Basic cleanup
# Ensure numeric
for col in ['reader_view','speed','dyslexia','dyslexia_bin']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Define dyslexia subsets
# Primary: dyslexia_bin == 1
# Sensitivity: dyslexia in {1,2}

sub_bin = df[df['dyslexia_bin'] == 1].copy()
sub_dx = df[df['dyslexia'].isin([1,2])].copy()


def summarize(sub):
    # group stats
    g = sub.groupby('reader_view')['speed']
    desc = g.agg(['count','mean','median','std']).reset_index()
    return desc


def cohen_d(x, y):
    # x, y arrays
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx-1)*sx + (ny-1)*sy) / (nx+ny-2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def run_tests(sub):
    # drop NaNs
    sub = sub[['reader_view','speed']].dropna()
    x = sub[sub['reader_view'] == 1]['speed'].values
    y = sub[sub['reader_view'] == 0]['speed'].values
    # Welch t-test
    t_res = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
    # Mann-Whitney U (two-sided)
    try:
        mw_res = stats.mannwhitneyu(x, y, alternative='two-sided')
    except ValueError:
        mw_res = None
    d = cohen_d(x, y)
    # log1p
    lx = np.log1p(x)
    ly = np.log1p(y)
    t_log = stats.ttest_ind(lx, ly, equal_var=False, nan_policy='omit')
    return {
        'n_reader_view_1': len(x),
        'n_reader_view_0': len(y),
        'mean_diff': np.mean(x) - np.mean(y),
        'median_diff': np.median(x) - np.median(y),
        't_stat': t_res.statistic,
        't_p': t_res.pvalue,
        'mw_u': None if mw_res is None else mw_res.statistic,
        'mw_p': None if mw_res is None else mw_res.pvalue,
        'cohen_d': d,
        't_log_stat': t_log.statistic,
        't_log_p': t_log.pvalue,
    }


def run_regression(sub, label):
    # Regression on log(speed), with reader_view and controls
    # Use robust SEs
    sub = sub.copy()
    sub = sub.dropna(subset=['speed','reader_view','page_id','device','num_words'])
    sub['log_speed'] = np.log1p(sub['speed'])
    # Keep numeric covariates where available
    # Use categorical for page_id, device, language
    formula = 'log_speed ~ reader_view + C(page_id) + num_words + C(device)'
    if 'age' in sub.columns:
        formula += ' + age'
    if 'gender' in sub.columns:
        formula += ' + C(gender)'
    if 'education' in sub.columns:
        formula += ' + C(education)'
    if 'language' in sub.columns:
        formula += ' + C(language)'
    if 'english_native' in sub.columns:
        formula += ' + C(english_native)'
    model = smf.ols(formula, data=sub).fit(cov_type='HC3')
    coef = model.params.get('reader_view', np.nan)
    pval = model.pvalues.get('reader_view', np.nan)
    return {
        'label': label,
        'n': len(sub),
        'coef_log_speed': coef,
        'p_log_speed': pval,
        'formula': formula,
    }


results = {
    'summary_bin': summarize(sub_bin).to_dict(orient='records'),
    'summary_dx': summarize(sub_dx).to_dict(orient='records'),
    'tests_bin': run_tests(sub_bin),
    'tests_dx': run_tests(sub_dx),
    'reg_bin': run_regression(sub_bin, 'dyslexia_bin==1'),
    'reg_dx': run_regression(sub_dx, 'dyslexia in {1,2}'),
}

print(json.dumps(results, indent=2))
