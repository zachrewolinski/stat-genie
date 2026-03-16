import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Dyslexic subset
if 'dyslexia_bin' in df.columns:
    dyslexic = df['dyslexia_bin'] == 1
else:
    dyslexic = df['dyslexia'].fillna(0) > 0

sub = df.loc[dyslexic].copy()
sub = sub.dropna(subset=['reader_view', 'speed'])
sub = sub[sub['speed'] > 0].copy()

# Create log speed
sub['log_speed'] = np.log(sub['speed'])

# Convert categoricals
for col in ['page_id', 'device', 'education', 'language', 'english_native', 'gender']:
    if col in sub.columns:
        sub[col] = sub[col].astype('category')

# Summary counts
summary = {
    'n_obs': int(sub.shape[0]),
    'n_participants': int(sub['uuid'].nunique()) if 'uuid' in sub.columns else None,
    'n_reader_view_1': int((sub['reader_view'] == 1).sum()),
    'n_reader_view_0': int((sub['reader_view'] == 0).sum()),
}

# Group means (raw and log)
means = sub.groupby('reader_view')['speed'].mean()
log_means = sub.groupby('reader_view')['log_speed'].mean()
summary['mean_speed_reader_view_1'] = float(means.get(1, np.nan))
summary['mean_speed_reader_view_0'] = float(means.get(0, np.nan))
summary['mean_log_speed_reader_view_1'] = float(log_means.get(1, np.nan))
summary['mean_log_speed_reader_view_0'] = float(log_means.get(0, np.nan))

# Effect size (Cohen's d) on log speed
rv1 = sub.loc[sub['reader_view'] == 1, 'log_speed']
rv0 = sub.loc[sub['reader_view'] == 0, 'log_speed']

if len(rv1) > 1 and len(rv0) > 1:
    s1 = rv1.std(ddof=1)
    s0 = rv0.std(ddof=1)
    sp = np.sqrt(((len(rv1)-1)*s1**2 + (len(rv0)-1)*s0**2) / (len(rv1)+len(rv0)-2))
    d = (rv1.mean() - rv0.mean()) / sp if sp > 0 else np.nan
else:
    d = np.nan
summary['cohens_d_log_speed'] = float(d) if np.isfinite(d) else None

# Nonparametric test
mw_p = None
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    mw_p = float(mw.pvalue)
except Exception:
    mw_p = None
summary['mannwhitney_p'] = mw_p

# Welch t-test on log speed
try:
    ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
    summary['welch_t_p'] = float(ttest.pvalue)
except Exception:
    summary['welch_t_p'] = None

# OLS with cluster-robust SE by participant
covariates = []
for c in ['page_id', 'device', 'num_words', 'Flesch_Kincaid', 'english_native', 'retake_trial']:
    if c in sub.columns:
        covariates.append(c)

formula = 'log_speed ~ reader_view'
if covariates:
    formula += ' + ' + ' + '.join(covariates)

model_cols = ['log_speed', 'reader_view'] + covariates
if 'uuid' in sub.columns:
    model_cols.append('uuid')

model_data = sub[model_cols].dropna().copy()

ols_result = None
ols_error = None
if model_data.shape[0] > 0:
    try:
        if 'uuid' in model_data.columns:
            ols_result = smf.ols(formula, model_data).fit(
                cov_type='cluster',
                cov_kwds={'groups': model_data['uuid']}
            )
        else:
            ols_result = smf.ols(formula, model_data).fit()
    except Exception as e:
        ols_error = str(e)

result = {
    'model': 'ols_cluster' if ols_result is not None else None,
    'coef_log_speed': None,
    'p_value': None,
    'percent_change': None,
    'n_model_obs': int(model_data.shape[0])
}

if ols_result is not None:
    coef = ols_result.params.get('reader_view', np.nan)
    pval = ols_result.pvalues.get('reader_view', np.nan)
    result['coef_log_speed'] = float(coef) if np.isfinite(coef) else None
    result['p_value'] = float(pval) if np.isfinite(pval) else None
    result['percent_change'] = float(np.exp(coef) - 1.0) if np.isfinite(coef) else None

output = {
    'summary': summary,
    'result': result,
    'ols_error': ols_error,
    'formula': formula,
}

print(json.dumps(output, indent=2))
