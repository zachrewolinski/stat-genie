import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Map variables
children = df['feature6']
affairs = df['feature2']

# Basic cleaning
# Ensure children categories standardized
children = children.astype(str).str.strip().str.lower()

# Outcome: extramarital affairs frequency (as given)
# Create binary indicator for any affair
any_affair = (affairs > 0).astype(int)

# Group stats
stats_by_child = df.assign(children=children, affairs=affairs, any_affair=any_affair).groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any_affair=('any_affair', 'mean')
)

# Welch t-test for difference in mean affairs
children_groups = {}
for grp in ['yes', 'no']:
    children_groups[grp] = affairs[children == grp]

# If any group missing, handle
if all(grp in children_groups and len(children_groups[grp]) > 1 for grp in ['yes','no']):
    t_stat, t_p = stats.ttest_ind(children_groups['yes'], children_groups['no'], equal_var=False)
else:
    t_stat, t_p = (np.nan, np.nan)

# Mann-Whitney U test (nonparametric)
if all(len(children_groups[grp]) > 0 for grp in ['yes','no']):
    u_stat, u_p = stats.mannwhitneyu(children_groups['yes'], children_groups['no'], alternative='two-sided')
else:
    u_stat, u_p = (np.nan, np.nan)

# Effect size: Cohen's d (yes - no)
if all(len(children_groups[grp]) > 1 for grp in ['yes','no']):
    mean_yes = children_groups['yes'].mean()
    mean_no = children_groups['no'].mean()
    var_yes = children_groups['yes'].var(ddof=1)
    var_no = children_groups['no'].var(ddof=1)
    n_yes = len(children_groups['yes'])
    n_no = len(children_groups['no'])
    pooled_sd = np.sqrt(((n_yes-1)*var_yes + (n_no-1)*var_no) / (n_yes+n_no-2))
    cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan
else:
    cohen_d = np.nan

# Difference in proportions for any affair
contingency = pd.crosstab(children, any_affair)
chi2, chi2_p, dof, exp = stats.chi2_contingency(contingency)

# Logistic regression for any affair ~ children
# children reference category: no
logit_model = smf.logit('any_affair ~ C(children)', data=df.assign(children=children, any_affair=any_affair)).fit(disp=0)
logit_params = logit_model.params
logit_conf = logit_model.conf_int()

# OLS on affairs frequency ~ children (unadjusted)
ols_model = smf.ols('affairs ~ C(children)', data=df.assign(children=children, affairs=affairs)).fit()

output = {
    'stats_by_child': stats_by_child.reset_index().to_dict(orient='records'),
    't_test': {'t_stat': t_stat, 'p_value': t_p},
    'mann_whitney': {'u_stat': u_stat, 'p_value': u_p},
    'cohen_d_yes_minus_no': cohen_d,
    'chi_square_any_affair': {'chi2': chi2, 'p_value': chi2_p},
    'logit_any_affair_params': logit_params.to_dict(),
    'logit_any_affair_conf_int': logit_conf.to_dict(),
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_model.pvalues.to_dict(),
}

print(json.dumps(output, indent=2))
