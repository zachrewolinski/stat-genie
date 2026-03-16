import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Focus on individuals with dyslexia (including severe)
_df = _df[_df['dyslexia_bin'] == 1].copy()

# Ensure speed positive for log transform
_df = _df[_df['speed'] > 0].copy()

# Group stats
rv1 = _df[_df['reader_view'] == 1]['speed'].dropna()
rv0 = _df[_df['reader_view'] == 0]['speed'].dropna()

n1, n0 = len(rv1), len(rv0)
mean1, mean0 = rv1.mean(), rv0.mean()
med1, med0 = rv1.median(), rv0.median()
std1, std0 = rv1.std(ddof=1), rv0.std(ddof=1)

# Welch t-test
if n1 > 1 and n0 > 1:
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    tstat, pval = np.nan, np.nan

# Effect size (Hedges g)
if n1 > 1 and n0 > 1:
    sp = np.sqrt(((n1 - 1) * std1**2 + (n0 - 1) * std0**2) / (n1 + n0 - 2))
    d = (mean1 - mean0) / sp if sp > 0 else np.nan
    J = 1 - 3 / (4 * (n1 + n0) - 9)
    g = d * J
else:
    d, g = np.nan, np.nan

# Percent difference in mean speed
pct_diff_mean = (mean1 - mean0) / mean0 if mean0 != 0 else np.nan
pct_diff_median = (med1 - med0) / med0 if med0 != 0 else np.nan

# Regression with controls and clustered SEs
_df['log_speed'] = np.log(_df['speed'])

formula = (
    "log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + C(gender) "
    "+ correct_rate + retake_trial"
)

# Drop rows with missing values in model variables to keep groups aligned
model_vars = [
    'log_speed', 'reader_view', 'num_words', 'page_id', 'device',
    'age', 'gender', 'correct_rate', 'retake_trial', 'uuid'
]
_df_model = _df[model_vars].dropna().copy()

model = smf.ols(formula, data=_df_model).fit(
    cov_type='cluster', cov_kwds={'groups': _df_model['uuid']}
)

beta = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
p_reg = model.pvalues.get('reader_view', np.nan)
ci_low, ci_high = model.conf_int().loc['reader_view'].tolist()

# Convert log effect to percent change
pct_change = np.exp(beta) - 1 if pd.notnull(beta) else np.nan

results = {
    "n_dyslexia": int(_df['uuid'].nunique()),
    "rows": int(_df.shape[0]),
    "n_rv1": int(n1),
    "n_rv0": int(n0),
    "mean_speed_rv1": float(mean1),
    "mean_speed_rv0": float(mean0),
    "median_speed_rv1": float(med1),
    "median_speed_rv0": float(med0),
    "std_speed_rv1": float(std1),
    "std_speed_rv0": float(std0),
    "tstat": float(tstat) if pd.notnull(tstat) else None,
    "pval": float(pval) if pd.notnull(pval) else None,
    "hedges_g": float(g) if pd.notnull(g) else None,
    "pct_diff_mean": float(pct_diff_mean) if pd.notnull(pct_diff_mean) else None,
    "pct_diff_median": float(pct_diff_median) if pd.notnull(pct_diff_median) else None,
    "reg_beta_log": float(beta) if pd.notnull(beta) else None,
    "reg_se": float(se) if pd.notnull(se) else None,
    "reg_p": float(p_reg) if pd.notnull(p_reg) else None,
    "reg_ci_low": float(ci_low) if pd.notnull(ci_low) else None,
    "reg_ci_high": float(ci_high) if pd.notnull(ci_high) else None,
    "reg_pct_change": float(pct_change) if pd.notnull(pct_change) else None,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
