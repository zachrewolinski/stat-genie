import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'reading.csv'

# Load data
_df = pd.read_csv(DATA_PATH)

# Ensure relevant columns
required_cols = ['reader_view', 'speed', 'dyslexia', 'dyslexia_bin', 'uuid', 'page_id', 'num_words', 'Flesch_Kincaid', 'device', 'language', 'age', 'gender', 'education', 'english_native']
missing_cols = [c for c in required_cols if c not in _df.columns]

# Basic cleaning
# Use dyslexia_bin if present; otherwise dyslexia >=1
if 'dyslexia_bin' in _df.columns:
    dyslexic_mask = _df['dyslexia_bin'] == 1
else:
    dyslexic_mask = _df['dyslexia'] >= 1

# Filter to dyslexic participants
_df_dys = _df.loc[dyslexic_mask].copy()

# Remove missing speed or reader_view
_df_dys = _df_dys.dropna(subset=['speed', 'reader_view'])

# Ensure binary reader_view
_df_dys = _df_dys[_df_dys['reader_view'].isin([0, 1])]

# Basic group stats
summary = _df_dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).reset_index()

# Effect size (Cohen's d) for two independent groups (Welch)
rv0 = _df_dys.loc[_df_dys['reader_view'] == 0, 'speed']
rv1 = _df_dys.loc[_df_dys['reader_view'] == 1, 'speed']

# Welch t-test
welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    mw_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    mw_u = None

# Cohen's d (pooled SD with unequal n)
# Using standard pooled SD (not assuming equal variances) is common but for Welch, use average of variances.
# We'll compute Hedges g with pooled SD using weighted variances.

def hedges_g(x, y):
    x = x.dropna()
    y = y.dropna()
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    # pooled SD
    s_pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_pooled == 0:
        return np.nan
    d = (x.mean() - y.mean()) / s_pooled
    # Hedges' correction
    correction = 1 - (3 / (4*(nx + ny) - 9))
    return d * correction

hedges_g_val = hedges_g(rv1, rv0)

# Log-transform regression to reduce skew
_df_dys = _df_dys.copy()
_df_dys['log_speed'] = np.log1p(_df_dys['speed'])

# Build regression model with covariates if available
# Include reader_view plus page-level controls.
# Use categorical for page_id, device, language to control for content/device differences.

formula_parts = ['reader_view']

if 'num_words' in _df_dys.columns:
    formula_parts.append('num_words')
if 'Flesch_Kincaid' in _df_dys.columns:
    formula_parts.append('Flesch_Kincaid')
if 'page_id' in _df_dys.columns:
    formula_parts.append('C(page_id)')
if 'device' in _df_dys.columns:
    formula_parts.append('C(device)')
if 'language' in _df_dys.columns:
    formula_parts.append('C(language)')
if 'age' in _df_dys.columns:
    formula_parts.append('age')
if 'gender' in _df_dys.columns:
    formula_parts.append('C(gender)')
if 'education' in _df_dys.columns:
    formula_parts.append('C(education)')
if 'english_native' in _df_dys.columns:
    formula_parts.append('C(english_native)')

formula = 'log_speed ~ ' + ' + '.join(formula_parts)

# Fit OLS with robust standard errors
model = None
model_summary = None
coef = None
pval = None

try:
    model = smf.ols(formula=formula, data=_df_dys).fit(cov_type='HC3')
    coef = model.params.get('reader_view', np.nan)
    pval = model.pvalues.get('reader_view', np.nan)
    model_summary = {
        'n': int(model.nobs),
        'coef': float(coef) if coef is not None else None,
        'pval': float(pval) if pval is not None else None,
        'r2': float(model.rsquared),
    }
except Exception as e:
    model_summary = {'error': str(e)}

# Also check within-subject if possible: participants with both conditions.
paired_results = None
if 'uuid' in _df_dys.columns:
    # Compute mean speed per participant per reader_view
    subj_means = _df_dys.groupby(['uuid', 'reader_view'])['speed'].mean().unstack('reader_view')
    # keep participants with both conditions
    paired = subj_means.dropna(subset=[0, 1])
    if len(paired) >= 5:
        # paired t-test on log speed to reduce skew
        paired_log = np.log1p(paired)
        t_res = stats.ttest_rel(paired_log[1], paired_log[0], nan_policy='omit')
        paired_results = {
            'n_pairs': int(len(paired)),
            'mean_diff_log': float((paired_log[1] - paired_log[0]).mean()),
            't_stat': float(t_res.statistic),
            'pval': float(t_res.pvalue),
        }
    else:
        paired_results = {'n_pairs': int(len(paired))}

output = {
    'missing_cols': missing_cols,
    'n_total': int(len(_df)),
    'n_dyslexic': int(len(_df_dys)),
    'summary_speed_by_reader_view': summary.to_dict(orient='records'),
    'welch_ttest': {'statistic': float(welch_t.statistic), 'pvalue': float(welch_t.pvalue)},
    'mannwhitneyu': None if mw_u is None else {'statistic': float(mw_u.statistic), 'pvalue': float(mw_u.pvalue)},
    'hedges_g_reader_view_minus_non': float(hedges_g_val),
    'regression_log_speed': model_summary,
    'paired_results': paired_results,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
