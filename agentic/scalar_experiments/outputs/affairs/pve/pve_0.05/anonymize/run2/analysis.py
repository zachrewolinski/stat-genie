import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Map variables
# feature2: extramarital affairs frequency (numeric)
# feature6: children in marriage (yes/no)

# Clean / encode
df = _df.copy()
# Ensure lower-case for children
df['children'] = df['feature6'].astype(str).str.strip().str.lower()
# Filter to yes/no
df = df[df['children'].isin(['yes', 'no'])].copy()

# outcome
y = df['feature2'].astype(float)

# group stats
stats_by = df.groupby('children')['feature2'].agg(['count', 'mean', 'median', 'std'])

# t-test (Welch)
no_vals = df.loc[df['children'] == 'no', 'feature2'].astype(float)
yes_vals = df.loc[df['children'] == 'yes', 'feature2'].astype(float)

ttest = stats.ttest_ind(no_vals, yes_vals, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    mw = stats.mannwhitneyu(no_vals, yes_vals, alternative='two-sided')
except Exception:
    mw = None

# Cohen's d
# Pooled SD with unequal sizes
n1, n2 = no_vals.shape[0], yes_vals.shape[0]
var1, var2 = no_vals.var(ddof=1), yes_vals.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohens_d = (no_vals.mean() - yes_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan

# Regression: feature2 ~ children (binary)
df['children_yes'] = (df['children'] == 'yes').astype(int)

X_simple = sm.add_constant(df['children_yes'])
model_simple = sm.OLS(y, X_simple).fit(cov_type='HC3')

# Regression with controls
# feature3 gender (female/male)
# feature4 age
# feature5 years married
# feature7 religiousness
# feature8 education
# feature9 occupation
# feature10 marriage rating

df['male'] = (df['feature3'].astype(str).str.strip().str.lower() == 'male').astype(int)

controls = ['children_yes', 'male', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']
X_ctrl = sm.add_constant(df[controls].astype(float))
model_ctrl = sm.OLS(y, X_ctrl).fit(cov_type='HC3')

# Also log1p outcome to reduce skew
logy = np.log1p(y)
model_simple_log = sm.OLS(logy, X_simple).fit(cov_type='HC3')
model_ctrl_log = sm.OLS(logy, X_ctrl).fit(cov_type='HC3')

# Collect results
results = {
    'group_stats': stats_by.to_dict(),
    'ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitney': None if mw is None else {'statistic': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'cohens_d_no_minus_yes': float(cohens_d),
    'reg_simple': {
        'coef_children_yes': float(model_simple.params['children_yes']),
        'p_children_yes': float(model_simple.pvalues['children_yes']),
        'n': int(model_simple.nobs),
        'r2': float(model_simple.rsquared),
    },
    'reg_ctrl': {
        'coef_children_yes': float(model_ctrl.params['children_yes']),
        'p_children_yes': float(model_ctrl.pvalues['children_yes']),
        'n': int(model_ctrl.nobs),
        'r2': float(model_ctrl.rsquared),
    },
    'reg_simple_log': {
        'coef_children_yes': float(model_simple_log.params['children_yes']),
        'p_children_yes': float(model_simple_log.pvalues['children_yes']),
        'n': int(model_simple_log.nobs),
        'r2': float(model_simple_log.rsquared),
    },
    'reg_ctrl_log': {
        'coef_children_yes': float(model_ctrl_log.params['children_yes']),
        'p_children_yes': float(model_ctrl_log.pvalues['children_yes']),
        'n': int(model_ctrl_log.nobs),
        'r2': float(model_ctrl_log.rsquared),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
