import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Identify dyslexia subset
if 'dyslexia_bin' in df.columns:
    dys_mask = df['dyslexia_bin'] == 1
elif 'dyslexia' in df.columns:
    dys_mask = df['dyslexia'] > 0
else:
    raise ValueError('No dyslexia indicator found')

df_dys = df.loc[dys_mask].copy()

# Keep required columns
req_cols = ['speed', 'reader_view', 'uuid']
for col in req_cols:
    if col not in df_dys.columns:
        raise ValueError(f'Missing column: {col}')

# Drop missing/invalid
clean = df_dys.dropna(subset=['speed', 'reader_view', 'uuid']).copy()
clean = clean[clean['speed'] > 0]

# Ensure reader_view is binary 0/1
clean = clean[clean['reader_view'].isin([0, 1])]

# Descriptives
summary = clean.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).reset_index()

# Log transform to reduce skew
clean['log_speed'] = np.log(clean['speed'])

# Welch t-test on log speed
log0 = clean.loc[clean['reader_view'] == 0, 'log_speed']
log1 = clean.loc[clean['reader_view'] == 1, 'log_speed']

t_stat, t_p = stats.ttest_ind(log1, log0, equal_var=False, nan_policy='omit')

# Mann-Whitney U on speed
u_stat, u_p = stats.mannwhitneyu(
    clean.loc[clean['reader_view'] == 1, 'speed'],
    clean.loc[clean['reader_view'] == 0, 'speed'],
    alternative='two-sided'
)

# Cohen's d on log speed
n1, n0 = log1.size, log0.size
s1, s0 = log1.std(ddof=1), log0.std(ddof=1)
pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
cohen_d = (log1.mean() - log0.mean()) / pooled if pooled > 0 else np.nan

# Cluster-robust OLS
ols = smf.ols('log_speed ~ reader_view + C(page_id)', data=clean).fit(
    cov_type='cluster', cov_kwds={'groups': clean['uuid']}
)

ols_beta = ols.params.get('reader_view', np.nan)
ols_p = ols.pvalues.get('reader_view', np.nan)

# Mixed effects model (random intercept for participant)
mixed_beta = np.nan
mixed_p = np.nan
mixed_ok = False
try:
    model = smf.mixedlm('log_speed ~ reader_view + C(page_id)', data=clean, groups=clean['uuid'])
    mfit = model.fit(reml=False)
    mixed_beta = mfit.params.get('reader_view', np.nan)
    mixed_p = mfit.pvalues.get('reader_view', np.nan)
    mixed_ok = True
except Exception:
    mixed_ok = False

# Percent change interpretation from OLS beta
pct_change = np.exp(ols_beta) - 1 if np.isfinite(ols_beta) else np.nan

result = {
    'n_total_dyslexia': int(clean.shape[0]),
    'n_participants': int(clean['uuid'].nunique()),
    'summary_by_reader_view': summary.to_dict(orient='records'),
    'welch_t_log_speed_p': float(t_p),
    'welch_t_log_speed_t': float(t_stat),
    'mannwhitney_p': float(u_p),
    'cohen_d_log_speed': float(cohen_d),
    'ols_beta_reader_view_log': float(ols_beta),
    'ols_p_reader_view': float(ols_p),
    'ols_pct_change': float(pct_change),
    'mixedlm_beta_reader_view_log': float(mixed_beta),
    'mixedlm_p_reader_view': float(mixed_p),
    'mixedlm_converged': bool(mixed_ok),
}

print(json.dumps(result, indent=2))
