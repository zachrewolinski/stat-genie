import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import json

# Load data
_df = pd.read_csv('reading.csv')

# Focus on dyslexic participants (dyslexia_bin == 1)
_df = _df[_df['dyslexia_bin'] == 1].copy()

# Basic counts (before dropping missing)
n_rows = len(_df)
unique_participants = _df['uuid'].nunique()

# Descriptive stats by reader_view
_group = _df.groupby('reader_view')['speed']
means = _group.mean()
medians = _group.median()
stds = _group.std()
counts = _group.size()

# Effect size on raw speed (Cohen's d for independent groups)
if counts.min() > 1:
    mean0, mean1 = means.get(0, np.nan), means.get(1, np.nan)
    std0, std1 = stds.get(0, np.nan), stds.get(1, np.nan)
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    pooled_std = np.sqrt(((n0-1)*std0**2 + (n1-1)*std1**2) / (n0+n1-2))
    cohen_d = (mean1 - mean0) / pooled_std if pooled_std > 0 else np.nan
else:
    cohen_d = np.nan

# Two-sample t-test (Welch) on raw speed
if counts.min() > 1:
    speeds0 = _df.loc[_df['reader_view'] == 0, 'speed']
    speeds1 = _df.loc[_df['reader_view'] == 1, 'speed']
    t_stat, t_p = stats.ttest_ind(speeds1, speeds0, equal_var=False, nan_policy='omit')
else:
    t_stat, t_p = np.nan, np.nan

# Within-subject analysis for participants who saw both conditions
pivot = _df.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna()
paired_n = len(paired)
if paired_n > 1:
    paired_t, paired_p = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    paired_mean_diff = (paired[1] - paired[0]).mean()
else:
    paired_t, paired_p, paired_mean_diff = np.nan, np.nan, np.nan

# Regression with random intercept for participant (MixedLM) on log(speed)
# Drop rows with missing values for variables used in the model
_df['log_speed'] = np.log(_df['speed'] + 1)
model_cols = ['log_speed', 'reader_view', 'page_id', 'num_words', 'Flesch_Kincaid', 'device', 'uuid']
_df_model = _df[model_cols].dropna().copy()

formula = 'log_speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid + C(device)'

mixed_result = None
try:
    mixed_model = smf.mixedlm(formula, _df_model, groups=_df_model['uuid'])
    mixed_result = mixed_model.fit(reml=False, method='lbfgs')
except Exception as e:
    mixed_result = e

# Cluster-robust OLS for comparison
ols_result = None
try:
    ols_model = smf.ols(formula, _df_model).fit(cov_type='cluster', cov_kwds={'groups': _df_model['uuid']})
    ols_result = ols_model
except Exception as e:
    ols_result = e

output = {
    'n_rows': n_rows,
    'unique_participants': unique_participants,
    'counts': counts.to_dict(),
    'mean_speed': means.to_dict(),
    'median_speed': medians.to_dict(),
    'std_speed': stds.to_dict(),
    'cohen_d_reader_view_effect_raw': cohen_d,
    'welch_ttest_t': t_stat,
    'welch_ttest_p': t_p,
    'paired_n': paired_n,
    'paired_t': paired_t,
    'paired_p': paired_p,
    'paired_mean_diff': paired_mean_diff,
    'model_rows': len(_df_model),
}

if not isinstance(mixed_result, Exception):
    output['mixedlm_reader_view_coef'] = mixed_result.params.get('reader_view', np.nan)
    output['mixedlm_reader_view_p'] = mixed_result.pvalues.get('reader_view', np.nan)
else:
    output['mixedlm_error'] = str(mixed_result)

if not isinstance(ols_result, Exception):
    output['ols_reader_view_coef'] = ols_result.params.get('reader_view', np.nan)
    output['ols_reader_view_p'] = ols_result.pvalues.get('reader_view', np.nan)
else:
    output['ols_error'] = str(ols_result)

print(json.dumps(output, indent=2, default=str))
