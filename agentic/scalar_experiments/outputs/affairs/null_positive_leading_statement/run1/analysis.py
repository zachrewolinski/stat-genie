import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic group stats
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Mean affairs by children
mean_by_children = _df.groupby('children')['affairs'].mean()
prop_any_by_children = _df.groupby('children')['any_affair'].mean()

# OLS on affairs counts with children indicator
_df['children_yes'] = (_df['children'] == 'yes').astype(int)
ols_model = smf.ols('affairs ~ children_yes', data=_df).fit()

# Logistic regression on any affair
logit_model = smf.logit('any_affair ~ children_yes', data=_df).fit(disp=False)

# Also adjust for potential confounders (age, yearsmarried, religiousness, education, occupation, rating, gender)
adj_formula = 'affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)'
adj_ols = smf.ols(adj_formula, data=_df).fit()

adj_logit_formula = 'any_affair ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)'
adj_logit = smf.logit(adj_logit_formula, data=_df).fit(disp=False)

results = {
    'mean_by_children': mean_by_children.to_dict(),
    'prop_any_by_children': prop_any_by_children.to_dict(),
    'ols_coef_children': float(ols_model.params['children_yes']),
    'ols_p_children': float(ols_model.pvalues['children_yes']),
    'logit_coef_children': float(logit_model.params['children_yes']),
    'logit_p_children': float(logit_model.pvalues['children_yes']),
    'adj_ols_coef_children': float(adj_ols.params['children_yes']),
    'adj_ols_p_children': float(adj_ols.pvalues['children_yes']),
    'adj_logit_coef_children': float(adj_logit.params['children_yes']),
    'adj_logit_p_children': float(adj_logit.pvalues['children_yes']),
    'n': int(len(_df))
}

# Save results to a file for inspection
pd.Series(results).to_json('analysis_results.json', indent=2)

print('Analysis complete. Results saved to analysis_results.json')
