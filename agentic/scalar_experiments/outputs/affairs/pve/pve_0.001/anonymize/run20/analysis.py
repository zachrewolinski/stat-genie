import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
outcome = 'feature2'
children_col = 'feature6'

# Clean/prepare
_df = _df.copy()
_df[children_col] = _df[children_col].astype(str).str.lower()
_df = _df[_df[children_col].isin(['yes', 'no'])]

# Group stats
stats_rows = []
for grp, sub in _df.groupby(children_col):
    vals = sub[outcome].astype(float)
    stats_rows.append({
        'children': grp,
        'n': int(vals.shape[0]),
        'mean': float(vals.mean()),
        'median': float(vals.median()),
        'std': float(vals.std(ddof=1))
    })

# Two-sample tests
vals_yes = _df[_df[children_col] == 'yes'][outcome].astype(float)
vals_no = _df[_df[children_col] == 'no'][outcome].astype(float)

ttest = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    mwu = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
    mwu_stat, mwu_p = float(mwu.statistic), float(mwu.pvalue)
except Exception:
    mwu_stat, mwu_p = None, None

# Effect size: Cohen's d (yes - no)
mean_yes, mean_no = vals_yes.mean(), vals_no.mean()
std_yes, std_no = vals_yes.std(ddof=1), vals_no.std(ddof=1)
pooled_sd = np.sqrt(((vals_yes.shape[0]-1)*std_yes**2 + (vals_no.shape[0]-1)*std_no**2) / (vals_yes.shape[0]+vals_no.shape[0]-2))
cohen_d = float((mean_yes - mean_no) / pooled_sd) if pooled_sd > 0 else np.nan

# OLS regression with controls
# Controls from metadata
# feature3: gender, feature4: age, feature5: years married, feature7: relig, feature8: education,
# feature9: occupation, feature10: marriage rating
formula = (
    'feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
)
ols = smf.ols(formula, data=_df).fit(cov_type='HC3')

# Logistic regression on any positive affairs (proxy)
_df['any_affair'] = (_df[outcome] > 0).astype(int)
logit = smf.logit('any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=_df).fit(disp=False)

# Extract coefficient for children (yes vs no)
# In statsmodels, C(feature6)[T.yes] uses 'no' as baseline if present.
ols_coef = ols.params.get('C(feature6)[T.yes]', np.nan)
ols_p = ols.pvalues.get('C(feature6)[T.yes]', np.nan)

logit_coef = logit.params.get('C(feature6)[T.yes]', np.nan)
logit_p = logit.pvalues.get('C(feature6)[T.yes]', np.nan)

results = {
    'group_stats': stats_rows,
    'ttest': {'stat': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitneyu': {'stat': mwu_stat, 'pvalue': mwu_p},
    'cohen_d_yes_minus_no': cohen_d,
    'ols_children_yes_coef': float(ols_coef),
    'ols_children_yes_pvalue': float(ols_p),
    'logit_children_yes_coef': float(logit_coef),
    'logit_children_yes_pvalue': float(logit_p)
}

print(json.dumps(results, indent=2))
