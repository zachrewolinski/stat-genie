import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('reading.csv')

# Ensure columns exist
# Filter dyslexia: dyslexia_bin == 1 or dyslexia > 0
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] > 0].copy()

# Basic counts
n_total = len(df)
n_dys = len(dys)

# Check reader_view distribution
rv_counts = dys['reader_view'].value_counts(dropna=False)

# Remove nonpositive or missing speed
speed = dys['speed']
dys = dys[(speed.notna()) & (speed > 0)].copy()

# Summary stats by reader_view
summary = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# T-test (Welch) for raw speed
rv0 = dys[dys['reader_view'] == 0]['speed']
rv1 = dys[dys['reader_view'] == 1]['speed']

# If group sizes adequate
if len(rv0) > 1 and len(rv1) > 1:
    t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_res = None

# Cohen's d
if len(rv0) > 1 and len(rv1) > 1:
    s0 = rv0.std(ddof=1)
    s1 = rv1.std(ddof=1)
    n0 = len(rv0)
    n1 = len(rv1)
    s_pooled = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    d = (rv1.mean() - rv0.mean()) / s_pooled if s_pooled > 0 else np.nan
else:
    d = np.nan

# Mixed effects model (log speed) with random intercept for uuid and page_id
# Add small constant to avoid log(0)
dys['log_speed'] = np.log(dys['speed'])

# Basic OLS with clustered SE by uuid
ols = smf.ols('log_speed ~ reader_view', data=dys).fit(cov_type='cluster', cov_kwds={'groups': dys['uuid']})

# MixedLM: random intercept for uuid, optional page_id if enough data
mixed = None
try:
    # Use uuid random intercept
    mixed = smf.mixedlm('log_speed ~ reader_view', data=dys, groups=dys['uuid']).fit(reml=False)
except Exception as e:
    mixed = None

# Additional model controlling for page_id (as fixed effect) and device, age
controls = []
for col in ['page_id','device','age','education','gender','language','english_native','retake_trial']:
    if col in dys.columns:
        controls.append(col)

# Build formula with categorical controls
formula = 'log_speed ~ reader_view'
for col in controls:
    if dys[col].dtype == 'object' or str(dys[col].dtype).startswith('category'):
        formula += f' + C({col})'
    else:
        formula += f' + {col}'

if controls:
    subset_cols = ['log_speed', 'reader_view', 'uuid'] + controls
    dys_controls = dys.dropna(subset=subset_cols).copy()
else:
    dys_controls = dys.copy()

ols_controls = smf.ols(formula, data=dys_controls).fit(
    cov_type='cluster',
    cov_kwds={'groups': dys_controls['uuid']}
)

# Output summary stats
print('n_total', n_total)
print('n_dys', n_dys)
print('rv_counts')
print(rv_counts)
print('summary')
print(summary)

if t_res is not None:
    print('welch_t', t_res)
print('cohens_d', d)

print('ols_coef', ols.params.get('reader_view', np.nan), 'p', ols.pvalues.get('reader_view', np.nan))
if mixed is not None:
    print('mixed_coef', mixed.params.get('reader_view', np.nan), 'p', mixed.pvalues.get('reader_view', np.nan))
print('ols_controls_coef', ols_controls.params.get('reader_view', np.nan), 'p', ols_controls.pvalues.get('reader_view', np.nan))

# Save some stats to csv for later if needed
summary.to_csv('summary_by_reader_view.csv')
