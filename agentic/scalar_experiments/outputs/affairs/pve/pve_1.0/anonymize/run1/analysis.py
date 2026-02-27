import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Identify key columns
# feature2: engagement in extramarital affairs (numeric)
# feature6: children in marriage (yes/no)

# Basic checks
col_affairs = 'feature2'
col_children = 'feature6'

# Clean: ensure children is categorical yes/no
_df[col_children] = _df[col_children].astype(str).str.lower().str.strip()

# Split groups
no_child = _df[_df[col_children] == 'no'][col_affairs]
yes_child = _df[_df[col_children] == 'yes'][col_affairs]

# Summary stats
summary = {
    'n_total': len(_df),
    'n_yes_children': len(yes_child),
    'n_no_children': len(no_child),
    'mean_yes_children': float(yes_child.mean()),
    'mean_no_children': float(no_child.mean()),
    'median_yes_children': float(yes_child.median()),
    'median_no_children': float(no_child.median()),
    'std_yes_children': float(yes_child.std(ddof=1)),
    'std_no_children': float(no_child.std(ddof=1)),
    'unique_affairs_values': int(_df[col_affairs].nunique()),
    'affairs_value_counts': _df[col_affairs].value_counts().to_dict(),
}

# Difference in means (Welch t-test)
welch = stats.ttest_ind(yes_child, no_child, equal_var=False, nan_policy='omit')
summary['welch_t_stat'] = float(welch.statistic)
summary['welch_p_value'] = float(welch.pvalue)
summary['mean_diff_yes_minus_no'] = float(summary['mean_yes_children'] - summary['mean_no_children'])

# Nonparametric test
mw = stats.mannwhitneyu(yes_child, no_child, alternative='two-sided')
summary['mw_u_stat'] = float(mw.statistic)
summary['mw_p_value'] = float(mw.pvalue)

# Effect size (Cohen's d for independent samples)
def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    pooled = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    if pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)

summary['cohens_d_yes_minus_no'] = float(cohens_d(yes_child, no_child))

# Regression with controls (OLS)
# Use other numeric features: feature3 (gender) categorical, feature4-5,7-10 numeric
_df['children_yes'] = (_df[col_children] == 'yes').astype(int)
_df['gender_male'] = (_df['feature3'].astype(str).str.lower().str.strip() == 'male').astype(int)

# Build model
# OLS: affairs ~ children + gender + age + years married + religiousness + education + occupation + marriage rating
formula = (
    'feature2 ~ children_yes + gender_male + feature4 + feature5 + '
    'feature7 + feature8 + feature9 + feature10'
)

model = smf.ols(formula=formula, data=_df).fit(cov_type='HC3')

summary['ols_children_coef'] = float(model.params.get('children_yes', np.nan))
summary['ols_children_p_value'] = float(model.pvalues.get('children_yes', np.nan))
summary['ols_r_squared'] = float(model.rsquared)

# Save summary
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
