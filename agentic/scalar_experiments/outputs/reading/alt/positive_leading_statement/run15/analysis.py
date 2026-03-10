import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Filter to dyslexia (binary indicator)
if 'dyslexia_bin' in _df.columns:
    df = _df[_df['dyslexia_bin'] == 1].copy()
else:
    df = _df[_df['dyslexia'] > 0].copy()

# Basic cleaning: drop missing speed or reader_view
df = df.dropna(subset=['speed', 'reader_view'])

# Group stats
grp = df.groupby('reader_view')['speed']
summary = grp.agg(['count', 'mean', 'median', 'std']).reset_index()

# Welch t-test
speed_rv1 = df[df['reader_view'] == 1]['speed']
speed_rv0 = df[df['reader_view'] == 0]['speed']

t_stat, p_val = stats.ttest_ind(speed_rv1, speed_rv0, equal_var=False, nan_policy='omit')

# Cohen's d (Hedges g) for unequal n
n1, n0 = len(speed_rv1), len(speed_rv0)
mean1, mean0 = speed_rv1.mean(), speed_rv0.mean()
std1, std0 = speed_rv1.std(ddof=1), speed_rv0.std(ddof=1)
# pooled sd (unbiased) for d
s_pooled = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2) / (n1 + n0 - 2)) if (n1 + n0 - 2) > 0 else np.nan
cohens_d = (mean1 - mean0) / s_pooled if np.isfinite(s_pooled) and s_pooled > 0 else np.nan
# Hedges g
J = 1 - (3/(4*(n1+n0)-9)) if (n1+n0) > 2 else 1
hedges_g = cohens_d * J if np.isfinite(cohens_d) else np.nan

# 95% CI for mean difference using Welch
mean_diff = mean1 - mean0
# Welch-Satterthwaite df
s1 = std1**2 / n1 if n1 > 0 else np.nan
s0 = std0**2 / n0 if n0 > 0 else np.nan
welch_df = (s1 + s0)**2 / ((s1**2)/(n1-1) + (s0**2)/(n0-1)) if n1 > 1 and n0 > 1 else np.nan
alpha = 0.05
if np.isfinite(welch_df):
    t_crit = stats.t.ppf(1 - alpha/2, welch_df)
    ci_low = mean_diff - t_crit * np.sqrt(s1 + s0)
    ci_high = mean_diff + t_crit * np.sqrt(s1 + s0)
else:
    ci_low = np.nan
    ci_high = np.nan

# Regression with covariates; cluster-robust SE by participant if possible
base_covs = []
for col in ['num_words', 'Flesch_Kincaid', 'page_id']:
    if col in df.columns:
        base_covs.append(col)

if base_covs:
    formula = 'speed ~ reader_view + ' + ' + '.join(base_covs)
else:
    formula = 'speed ~ reader_view'

# Use OLS with cluster-robust SE by uuid if available
model = smf.ols(formula, data=df).fit()
if 'uuid' in df.columns:
    try:
        model_robust = model.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
    except Exception:
        model_robust = model
else:
    model_robust = model

# Helper to extract param by name
param_names = list(getattr(model_robust, 'model', model).exog_names)
params = np.asarray(model_robust.params)
bses = np.asarray(model_robust.bse)
if isinstance(model_robust.params, pd.Series):
    params_series = model_robust.params
    bse_series = model_robust.bse
else:
    params_series = pd.Series(params, index=param_names)
    bse_series = pd.Series(bses, index=param_names)

coef_rv = params_series.get('reader_view', np.nan)
se_rv = bse_series.get('reader_view', np.nan)

try:
    p_rv = model_robust.pvalues['reader_view']
except Exception:
    # fallback: use tvalues if available
    try:
        tval = model_robust.tvalues[param_names.index('reader_view')]
        df_resid = model_robust.df_resid
        p_rv = 2 * stats.t.sf(np.abs(tval), df_resid)
    except Exception:
        p_rv = np.nan

# Save analysis results for inspection
out = {
    'n_total_dyslexia': int(len(df)),
    'group_summary': summary.to_dict(orient='records'),
    't_test': {
        't_stat': float(t_stat) if np.isfinite(t_stat) else None,
        'p_value': float(p_val) if np.isfinite(p_val) else None,
        'mean_diff': float(mean_diff) if np.isfinite(mean_diff) else None,
        'ci_low': float(ci_low) if np.isfinite(ci_low) else None,
        'ci_high': float(ci_high) if np.isfinite(ci_high) else None,
        'cohens_d': float(cohens_d) if np.isfinite(cohens_d) else None,
        'hedges_g': float(hedges_g) if np.isfinite(hedges_g) else None,
    },
    'regression': {
        'formula': formula,
        'coef_reader_view': float(coef_rv) if np.isfinite(coef_rv) else None,
        'se_reader_view': float(se_rv) if np.isfinite(se_rv) else None,
        'p_value_reader_view': float(p_rv) if np.isfinite(p_rv) else None,
        'n_obs': int(model_robust.nobs),
        'r2': float(model.rsquared) if hasattr(model, 'rsquared') else None,
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
