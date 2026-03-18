import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure feature6 is categorical with yes/no
if df['feature6'].dtype != 'object':
    df['feature6'] = df['feature6'].astype(str)

# Main variables
outcome = df['feature2']
children = df['feature6']

# Group stats
stats_by_group = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# Two-sample t-test (Welch)
child_yes = df.loc[df['feature6'] == 'yes', 'feature2']
child_no = df.loc[df['feature6'] == 'no', 'feature2']

# Drop NaNs just in case
child_yes = child_yes.dropna()
child_no = child_no.dropna()

# Welch t-test
t_stat, t_p = stats.ttest_ind(child_yes, child_no, equal_var=False)

# Mann-Whitney U test (non-parametric)
try:
    u_stat, u_p = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except Exception:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d (using pooled SD)
mean_yes = child_yes.mean()
mean_no = child_no.mean()
std_yes = child_yes.std(ddof=1)
std_no = child_no.std(ddof=1)

n_yes = child_yes.shape[0]
n_no = child_no.shape[0]
pooled_sd = np.sqrt(((n_yes - 1)*std_yes**2 + (n_no - 1)*std_no**2) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Regression: OLS with robust SE
# Control variables: age, years married, religiousness, education, occupation, marriage rating, gender
# feature3 is gender categorical

# Prepare data for regression
reg_df = df.copy()
reg_df['children_yes'] = (reg_df['feature6'] == 'yes').astype(int)

# OLS with controls
formula = 'feature2 ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)'
ols_model = smf.ols(formula, data=reg_df).fit(cov_type='HC3')

# Logistic regression for any affairs (feature2 > 0)
reg_df['any_affair'] = (reg_df['feature2'] > 0).astype(int)
logit_model = smf.logit('any_affair ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)', data=reg_df).fit(disp=False)

# Extract key results
ols_children_coef = ols_model.params['children_yes']
ols_children_p = ols_model.pvalues['children_yes']

logit_children_coef = logit_model.params['children_yes']
logit_children_p = logit_model.pvalues['children_yes']

# Convert logit coef to odds ratio
logit_or = np.exp(logit_children_coef)

results = {
    'n_total': int(df.shape[0]),
    'group_stats': stats_by_group.to_dict(),
    't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'mannwhitney': {'u_stat': float(u_stat), 'p_value': float(u_p)},
    'cohen_d': float(cohen_d),
    'ols_children': {'coef': float(ols_children_coef), 'p_value': float(ols_children_p)},
    'logit_children': {'coef': float(logit_children_coef), 'p_value': float(logit_children_p), 'odds_ratio': float(logit_or)},
}

print(json.dumps(results, indent=2))
