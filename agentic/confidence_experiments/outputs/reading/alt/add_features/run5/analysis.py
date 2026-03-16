import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Determine dyslexia indicator
if 'dyslexia_bin' in df.columns:
    df_dys = df[df['dyslexia_bin'] == 1].copy()
else:
    df_dys = df[df['dyslexia'] > 0].copy()

# Keep rows with speed and reader_view
cols_needed = ['speed', 'reader_view', 'uuid', 'page_id']
for c in cols_needed:
    if c not in df_dys.columns:
        raise ValueError(f"Missing column: {c}")

df_dys = df_dys[cols_needed].dropna()

# Ensure numeric
for c in ['speed', 'reader_view']:
    df_dys[c] = pd.to_numeric(df_dys[c], errors='coerce')

df_dys = df_dys.dropna(subset=['speed', 'reader_view'])

# Remove non-positive speed if any
if (df_dys['speed'] <= 0).any():
    df_dys = df_dys[df_dys['speed'] > 0]

# Summary stats
n_total = len(df_dys)
counts = df_dys['reader_view'].value_counts().to_dict()
mean_speed = df_dys.groupby('reader_view')['speed'].mean().to_dict()
median_speed = df_dys.groupby('reader_view')['speed'].median().to_dict()

# Log transform for analysis
# Add small constant not needed since speeds positive
log_speed = np.log(df_dys['speed'])
df_dys = df_dys.assign(log_speed=log_speed)

# Welch t-test on log speed
rv0 = df_dys[df_dys['reader_view'] == 0]['log_speed']
rv1 = df_dys[df_dys['reader_view'] == 1]['log_speed']

t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)

# Cohen's d (using pooled SD from Welch approximate)
# Use standard formula with pooled SD from group variances and sizes
n0, n1 = len(rv0), len(rv1)
var0, var1 = rv0.var(ddof=1), rv1.var(ddof=1)
# Pooled SD
pooled_sd = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else np.nan
cohens_d = (rv1.mean() - rv0.mean()) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Cluster-robust OLS with page fixed effects
# Some uuids might be missing; ensure category

# Keep only rows with page_id
df_model = df_dys.dropna(subset=['page_id', 'uuid']).copy()

# Convert to categorical
# If too many categories, statsmodels will handle

model = smf.ols('log_speed ~ reader_view + C(page_id)', data=df_model).fit(
    cov_type='cluster', cov_kwds={'groups': df_model['uuid']}
)

# Extract reader_view coefficient
coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
p_model = model.pvalues.get('reader_view', np.nan)

# Back-transform effect: exp(coef)-1 approximate percent change
pct_change = np.exp(coef) - 1 if np.isfinite(coef) else np.nan

# Print key results for manual review
print('n_total', n_total)
print('counts', counts)
print('mean_speed', mean_speed)
print('median_speed', median_speed)
print('t_stat', t_stat, 'p_val', p_val)
print('cohens_d', cohens_d)
print('coef', coef, 'se', se, 'p_model', p_model, 'pct_change', pct_change)
