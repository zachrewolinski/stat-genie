import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'reading.csv'
df = pd.read_csv(DF_PATH)

# Focus on participants with dyslexia (binary indicator)
df_d = df[df['dyslexia_bin'] == 1].copy()

# Exclude retake trials if indicated
if 'retake_trial' in df_d.columns:
    df_d = df_d[df_d['retake_trial'] == 0]

# Keep positive speeds
df_d = df_d[df_d['speed'] > 0].copy()

# Create log speed for modeling
# Add small epsilon not needed since speed>0
try:
    df_d['log_speed'] = np.log(df_d['speed'])
except Exception:
    df_d['log_speed'] = np.log(df_d['speed'].clip(lower=1e-6))

# Group summaries
summary = df_d.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Effect size (Hedges' g)
rv0 = df_d[df_d['reader_view'] == 0]['speed'].to_numpy()
rv1 = df_d[df_d['reader_view'] == 1]['speed'].to_numpy()

# Avoid division by zero
n0, n1 = len(rv0), len(rv1)
mean0, mean1 = rv0.mean(), rv1.mean()
var0, var1 = rv0.var(ddof=1), rv1.var(ddof=1)
pooled_sd = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else np.nan
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan
# Hedges' g correction
J = 1 - (3 / (4 * (n0 + n1) - 9)) if (n0 + n1) > 2 else np.nan
hedges_g = cohens_d * J if np.isfinite(cohens_d) and np.isfinite(J) else np.nan

# Statistical tests
# Welch t-test on log speed
log0 = df_d[df_d['reader_view'] == 0]['log_speed']
log1 = df_d[df_d['reader_view'] == 1]['log_speed']

ttest_log = stats.ttest_ind(log1, log0, equal_var=False, nan_policy='omit')

# Mann-Whitney U on raw speed (two-sided)
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except Exception:
    mw = None

# Mixed-effects model with random intercept for participant
# Keep model parsimonious to aid convergence
# Use page_id and device as fixed effects; age if present
fixed_parts = ['reader_view', 'C(page_id)', 'C(device)']
if 'age' in df_d.columns:
    fixed_parts.append('age')
formula = 'log_speed ~ ' + ' + '.join(fixed_parts)

mixed_result = None
ols_result = None
try:
    # Drop rows with missing modeling fields and reset index to avoid internal indexing issues
    model_cols = ['log_speed', 'reader_view', 'page_id', 'device', 'uuid']
    if 'age' in df_d.columns:
        model_cols.append('age')
    model_df = df_d[model_cols].dropna().reset_index(drop=True)
    mixed_model = smf.mixedlm(formula, model_df, groups=model_df['uuid'])
    mixed_result = mixed_model.fit(reml=False, method='lbfgs', maxiter=200, disp=False)
except Exception as e:
    mixed_result = e

# Cluster-robust OLS as a fallback/robustness check
try:
    if 'model_df' in locals() and len(model_df) > 0:
        ols_model = smf.ols(formula, data=model_df)
        ols_result = ols_model.fit(cov_type='cluster', cov_kwds={'groups': model_df['uuid']})
except Exception as e:
    ols_result = e

# Print outputs for inspection
print('Dyslexia subset rows:', len(df_d))
print('\nGroup summary (speed):')
print(summary)
print('\nEffect size (Hedges g):', hedges_g)
print('\nWelch t-test on log(speed):', ttest_log)
print('\nMann-Whitney U:', mw)
print('\nMixedLM formula:', formula)
print('\nMixedLM result:')
print(mixed_result)
print('\nCluster-robust OLS result:')
print(ols_result)
if hasattr(ols_result, 'params') and 'reader_view' in ols_result.params.index:
    coef = ols_result.params['reader_view']
    pval = ols_result.pvalues['reader_view']
    ci_low, ci_high = ols_result.conf_int().loc['reader_view']
    pct_change = (np.exp(coef) - 1) * 100
    pct_low = (np.exp(ci_low) - 1) * 100
    pct_high = (np.exp(ci_high) - 1) * 100
    print('\\nOLS reader_view effect on log(speed):')
    print('coef=', coef, 'p=', pval, '95% CI=', (ci_low, ci_high))
    print('approx % change in speed:', pct_change, '95% CI %=', (pct_low, pct_high))
