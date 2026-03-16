import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Define dyslexia group (includes dyslexia and severe dyslexia)
dys = df[df['dyslexia_bin'] == 1].copy()

# Basic counts
n_rows = len(dys)
n_participants = dys['uuid'].nunique()

# Reader view groups
rv0 = dys[dys['reader_view'] == 0]
rv1 = dys[dys['reader_view'] == 1]

# Handle non-positive speeds for log transform
# Speed should be positive, but guard just in case
valid = dys['speed'] > 0

dys = dys[valid].copy()
rv0 = dys[dys['reader_view'] == 0]
rv1 = dys[dys['reader_view'] == 1]

# Summary stats
summary = {}
for label, subset in [('no_reader_view', rv0), ('reader_view', rv1)]:
    summary[label] = {
        'n': len(subset),
        'mean_speed': subset['speed'].mean(),
        'median_speed': subset['speed'].median(),
        'std_speed': subset['speed'].std(),
    }

# Effect size (Cohen's d) on raw speed
mean0, mean1 = summary['no_reader_view']['mean_speed'], summary['reader_view']['mean_speed']
std0, std1 = summary['no_reader_view']['std_speed'], summary['reader_view']['std_speed']

# Pooled SD
pooled_sd = np.sqrt(((summary['no_reader_view']['n'] - 1) * std0 ** 2 + (summary['reader_view']['n'] - 1) * std1 ** 2) /
                    (summary['no_reader_view']['n'] + summary['reader_view']['n'] - 2))
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

# Parametric t-test (Welch) on log speed
log_speed = np.log(dys['speed'])
rv0_log = log_speed[dys['reader_view'] == 0]
rv1_log = log_speed[dys['reader_view'] == 1]

t_stat, p_val = stats.ttest_ind(rv1_log, rv0_log, equal_var=False)

# Non-parametric test (Mann-Whitney U) on raw speed
try:
    u_stat, u_p = stats.mannwhitneyu(rv1['speed'], rv0['speed'], alternative='two-sided')
except Exception:
    u_stat, u_p = np.nan, np.nan

# OLS with controls and cluster-robust SE by participant
# Use log speed to reduce skew
model_df = dys.copy()
model_df['log_speed'] = np.log(model_df['speed'])

# Build formula with key controls
formula = 'log_speed ~ reader_view + num_words + Flesch_Kincaid + C(device) + C(page_id) + age + C(gender) + C(english_native) + retake_trial'

# Drop rows with missing values in any model term to keep groups aligned
model_df = model_df.dropna(subset=[
    'log_speed', 'reader_view', 'num_words', 'Flesch_Kincaid', 'device', 'page_id',
    'age', 'gender', 'english_native', 'retake_trial', 'uuid'
]).copy()

ols_model = smf.ols(formula=formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['uuid']})

coef = ols_model.params.get('reader_view', np.nan)
se = ols_model.bse.get('reader_view', np.nan)
ols_p = ols_model.pvalues.get('reader_view', np.nan)

# Convert log coefficient to percent change
percent_change = (np.exp(coef) - 1) * 100 if pd.notnull(coef) else np.nan

# 95% CI for percent change
if pd.notnull(coef) and pd.notnull(se):
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    pct_ci_low = (np.exp(ci_low) - 1) * 100
    pct_ci_high = (np.exp(ci_high) - 1) * 100
else:
    pct_ci_low = np.nan
    pct_ci_high = np.nan

results = {
    'dyslexia_rows': int(n_rows),
    'dyslexia_participants': int(n_participants),
    'summary': summary,
    'cohens_d_raw': cohens_d,
    't_test_log_speed_p': p_val,
    't_test_log_speed_t': t_stat,
    'mannwhitney_p': u_p,
    'ols_reader_view_coef_log': coef,
    'ols_reader_view_se_log': se,
    'ols_reader_view_p': ols_p,
    'ols_reader_view_percent_change': percent_change,
    'ols_reader_view_percent_change_ci': [pct_ci_low, pct_ci_high],
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
