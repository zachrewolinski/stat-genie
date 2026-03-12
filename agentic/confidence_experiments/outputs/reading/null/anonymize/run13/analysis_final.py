import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def cohen_d(x, y):
    x = x.dropna()
    y = y.dropna()
    n1, n0 = len(x), len(y)
    if n1 < 2 or n0 < 2:
        return np.nan
    s1, s0 = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled


def analyze(df, speed_col):
    # Ensure positive speeds for log
    sub = df.copy()
    sub = sub[np.isfinite(sub[speed_col])]
    sub = sub[sub[speed_col] > 0]
    # groups
    rv = sub['feature3']
    g1 = sub.loc[rv == 1, speed_col]
    g0 = sub.loc[rv == 0, speed_col]

    res = {
        'n_total': int(len(sub)),
        'n_rv1': int(g1.count()),
        'n_rv0': int(g0.count()),
        'mean_rv1': float(g1.mean()),
        'mean_rv0': float(g0.mean()),
        'median_rv1': float(g1.median()),
        'median_rv0': float(g0.median()),
        'diff_mean': float(g1.mean() - g0.mean()),
        'diff_median': float(g1.median() - g0.median()),
        'cohen_d': float(cohen_d(g1, g0)) if g1.count() > 1 and g0.count() > 1 else np.nan,
    }

    # Welch t-test on raw speed
    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    res['t_pvalue'] = float(pval)

    # Mann-Whitney U test
    try:
        ustat, pval_u = stats.mannwhitneyu(g1, g0, alternative='two-sided')
    except ValueError:
        pval_u = np.nan
    res['mw_pvalue'] = float(pval_u) if np.isfinite(pval_u) else np.nan

    # Regression on log speed with controls
    sub = sub.copy()
    sub['log_speed'] = np.log(sub[speed_col])
    # Build model matrix
    X = sub[['feature3', 'feature7', 'feature16']].copy()
    X = pd.concat([
        X,
        pd.get_dummies(sub['feature2'], prefix='page', drop_first=True),
        pd.get_dummies(sub['feature11'], prefix='device', drop_first=True),
        pd.get_dummies(sub['feature15'], prefix='lang', drop_first=True),
    ], axis=1)
    X = sm.add_constant(X, has_constant='add')
    # Align and drop any rows with missing values
    data = pd.concat([sub['log_speed'], X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y = data['log_speed']
    X_clean = data.drop(columns=['log_speed'])
    if len(data) > 10:
        model = sm.OLS(y, X_clean).fit(cov_type='HC3')
        res['reg_coef_rv'] = float(model.params.get('feature3', np.nan))
        res['reg_pvalue_rv'] = float(model.pvalues.get('feature3', np.nan))
        res['reg_n'] = int(model.nobs)
    else:
        res['reg_coef_rv'] = np.nan
        res['reg_pvalue_rv'] = np.nan
        res['reg_n'] = int(len(data))
    return res


df = pd.read_csv('reading.csv')

# Derived speeds
# Words per minute based on time on page and time minus scrolling

df['speed_wpm_time'] = 60000 * df['feature7'] / df['feature4']
df['speed_wpm_read'] = 60000 * df['feature7'] / df['feature5']

results = {}

# dyslexia definitions
sub_dys17 = df[df['feature17'] == 1]
sub_dys12 = df[df['feature12'] > 0]

# Analyze three speed measures
speed_measures = {
    'feature20': 'feature20',
    'speed_wpm_time': 'speed_wpm_time',
    'speed_wpm_read': 'speed_wpm_read',
}

results['dyslexia_feature17'] = {k: analyze(sub_dys17, v) for k, v in speed_measures.items()}
results['dyslexia_feature12'] = {k: analyze(sub_dys12, v) for k, v in speed_measures.items()}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
