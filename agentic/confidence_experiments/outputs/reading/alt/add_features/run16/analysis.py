import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('reading.csv')

# ensure columns exist
required = ['uuid','reader_view','speed','dyslexia','dyslexia_bin']
missing = [c for c in required if c not in _df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# define dyslexia subset: dyslexia_bin==1 preferred; fallback dyslexia>=1
if _df['dyslexia_bin'].notna().any():
    dys = _df[_df['dyslexia_bin'] == 1].copy()
else:
    dys = _df[_df['dyslexia'] >= 1].copy()

# clean: drop missing speed/reader_view
for col in ['speed','reader_view']:
    dys = dys[pd.notna(dys[col])]

# reader_view as binary
# In case of float, round or cast

dys['reader_view'] = dys['reader_view'].astype(int)

# Basic group stats
stats_by_group = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# Unpaired t-test (Welch)
rv1 = dys.loc[dys['reader_view'] == 1, 'speed']
rv0 = dys.loc[dys['reader_view'] == 0, 'speed']

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (non-parametric)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    mwu = None

# Effect size: Cohen's d (unpaired)

def cohens_d(x, y):
    x = x.dropna()
    y = y.dropna()
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled) if pooled > 0 else np.nan

cd = cohens_d(rv1, rv0)

# Regression with robust SEs clustered by participant (uuid)
# include controls to reduce confounding: page_id, num_words, language, device, age
controls = ['page_id','num_words','language','device','age','gender','education','english_native','Flesch_Kincaid','correct_rate','retake_trial']

# Keep controls that exist
controls = [c for c in controls if c in dys.columns]

# Build design matrix using categorical dummies
X = dys[['reader_view'] + controls].copy()
X = pd.get_dummies(X, columns=[c for c in controls if dys[c].dtype == 'object' or str(dys[c].dtype).startswith('category')], drop_first=True)
X = sm.add_constant(X, has_constant='add')

# Outcome
Y = dys['speed']

# Drop rows with missing in X or Y
mask = X.notna().all(axis=1) & Y.notna()
X = X.loc[mask]
Y = Y.loc[mask]
clusters = dys.loc[mask, 'uuid']

# OLS with cluster-robust SEs if possible
model = sm.OLS(Y, X)
res = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

# Extract coefficient for reader_view
coef = res.params.get('reader_view', np.nan)
se = res.bse.get('reader_view', np.nan)
pval = res.pvalues.get('reader_view', np.nan)

# Summaries
summary = {
    'n_total_dys': int(len(dys)),
    'n_rv1': int((dys['reader_view'] == 1).sum()),
    'n_rv0': int((dys['reader_view'] == 0).sum()),
    'group_stats': stats_by_group.to_dict(),
    'ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mwu': None if mwu is None else {'statistic': float(mwu.statistic), 'pvalue': float(mwu.pvalue)},
    'cohens_d': float(cd),
    'regression_reader_view': {'coef': float(coef), 'se': float(se), 'pvalue': float(pval)},
    'controls_used': controls,
    'n_reg': int(len(Y)),
}

print(summary)
