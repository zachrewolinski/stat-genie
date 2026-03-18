import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

# Load data
file_path = 'affairs.csv'
df = pd.read_csv(file_path)

# Ensure expected columns
# feature2 = affair frequency
# feature6 = children yes/no

# Clean/encode
df['children_yes'] = (df['feature6'].str.lower() == 'yes').astype(int)

# Affair measures
# Any affair indicator
any_affair = (df['feature2'] > 0).astype(int)
df['any_affair'] = any_affair

# Group stats
stats_by_children = df.groupby('children_yes')['feature2'].agg(['mean','median','std','count'])
prop_any = df.groupby('children_yes')['any_affair'].mean()

# Mann-Whitney U test for distributions (nonparam)
no_children = df.loc[df['children_yes']==0, 'feature2']
children = df.loc[df['children_yes']==1, 'feature2']

# If all values? Use alternative two-sided
u_stat, u_p = stats.mannwhitneyu(children, no_children, alternative='two-sided')

# t-test (Welch) for mean difference
# Note: feature2 is not strictly continuous, but gives another view
_t_stat, t_p = stats.ttest_ind(children, no_children, equal_var=False)

# Difference in proportion any affair (z-test)
# using proportions test
from statsmodels.stats.proportion import proportions_ztest

count = np.array([df.loc[df['children_yes']==1, 'any_affair'].sum(),
                  df.loc[df['children_yes']==0, 'any_affair'].sum()])

nobs = np.array([df.loc[df['children_yes']==1, 'any_affair'].shape[0],
                 df.loc[df['children_yes']==0, 'any_affair'].shape[0]])

z_stat, z_p = proportions_ztest(count, nobs)

# Logistic regression for any affair with controls
# Controls: gender, age, years married, religiousness, education, occupation, marriage rating
# Encode gender (feature3)
# Some features are numeric already

# Prepare design matrix
X = df[['children_yes','feature4','feature5','feature7','feature8','feature9','feature10']].copy()
# Gender
X['male'] = (df['feature3'].str.lower()=='male').astype(int)

# Add constant
X = sm.add_constant(X)

logit_model = sm.Logit(df['any_affair'], X)
logit_res = logit_model.fit(disp=False)

# OLS for affair frequency with controls (for direction)
ols_model = sm.OLS(df['feature2'], X)
ols_res = ols_model.fit()

# Collect results
results = {
    'stats_by_children': stats_by_children.to_dict(),
    'prop_any': prop_any.to_dict(),
    'mannwhitney_u': {'u_stat': float(u_stat), 'p_value': float(u_p)},
    't_test': {'t_stat': float(_t_stat), 'p_value': float(t_p)},
    'prop_ztest': {'z_stat': float(z_stat), 'p_value': float(z_p)},
    'logit_children_coef': float(logit_res.params['children_yes']),
    'logit_children_p': float(logit_res.pvalues['children_yes']),
    'logit_children_odds_ratio': float(np.exp(logit_res.params['children_yes'])),
    'ols_children_coef': float(ols_res.params['children_yes']),
    'ols_children_p': float(ols_res.pvalues['children_yes']),
}

print(json.dumps(results, indent=2))
