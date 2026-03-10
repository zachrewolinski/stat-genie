import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Focus on dyslexia participants (binary indicator)
df = _df[_df['dyslexia_bin'] == 1].copy()

# Basic cleaning
# Remove non-positive speeds for log transform
speed = df['speed'].astype(float)
df = df[speed > 0].copy()

# Group stats
stats_by_view = df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Welch t-test on raw speed
speed_on = df[df['reader_view'] == 1]['speed']
speed_off = df[df['reader_view'] == 0]['speed']
welch_t = stats.ttest_ind(speed_on, speed_off, equal_var=False, nan_policy='omit')

# Effect size (Cohen's d) for raw speed
n1, n0 = speed_on.shape[0], speed_off.shape[0]
var1, var0 = speed_on.var(ddof=1), speed_off.var(ddof=1)
# Pooled SD for d (using weighted by n-1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)) if (n1 + n0 - 2) > 0 else np.nan
cohen_d = (speed_on.mean() - speed_off.mean()) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Log transform to address skew
log_speed = np.log(df['speed'])
df['log_speed'] = log_speed
log_on = df[df['reader_view'] == 1]['log_speed']
log_off = df[df['reader_view'] == 0]['log_speed']
welch_t_log = stats.ttest_ind(log_on, log_off, equal_var=False, nan_policy='omit')
var1_l, var0_l = log_on.var(ddof=1), log_off.var(ddof=1)
pooled_sd_l = np.sqrt(((n1 - 1) * var1_l + (n0 - 1) * var0_l) / (n1 + n0 - 2)) if (n1 + n0 - 2) > 0 else np.nan
cohen_d_log = (log_on.mean() - log_off.mean()) / pooled_sd_l if pooled_sd_l and pooled_sd_l > 0 else np.nan

# Non-parametric test
mannwhitney = stats.mannwhitneyu(speed_on, speed_off, alternative='two-sided')

# Regression with controls and clustered SE by participant
# Use a modest set of controls to avoid overfitting and singularity
formula = (
    'log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + C(gender) '
    '+ retake_trial + correct_rate + img_width + Flesch_Kincaid + C(english_native)'
)

# Ensure clustering groups align with model rows
model_cols = [
    'log_speed', 'reader_view', 'num_words', 'page_id', 'device', 'age', 'gender',
    'retake_trial', 'correct_rate', 'img_width', 'Flesch_Kincaid', 'english_native', 'uuid'
]
df_model = df[model_cols].dropna().copy()
model = smf.ols(formula=formula, data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['uuid']}
)
coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

results = {
    'n_dyslexia': int(df.shape[0]),
    'n_on': int(n1),
    'n_off': int(n0),
    'stats_by_view': stats_by_view.to_dict(),
    'welch_t_raw': {'stat': float(welch_t.statistic), 'p': float(welch_t.pvalue)},
    'cohen_d_raw': float(cohen_d),
    'welch_t_log': {'stat': float(welch_t_log.statistic), 'p': float(welch_t_log.pvalue)},
    'cohen_d_log': float(cohen_d_log),
    'mannwhitney': {'stat': float(mannwhitney.statistic), 'p': float(mannwhitney.pvalue)},
    'regression_reader_view': {'coef_log_speed': float(coef), 'p': float(pval)}
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
