import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Map children yes/no
children = df['feature6'].astype(str).str.lower().map({'yes':1,'no':0})
# outcome
y = df['feature2']

# group stats
stats_by = df.assign(children=children).groupby('children')['feature2'].agg(['count','mean','median','std'])

# t-test (Welch)
no = y[children==0]
yes = y[children==1]

t_res = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U
u_res = stats.mannwhitneyu(yes, no, alternative='two-sided')

# OLS regression with controls (age, years married, religiousness, education, occupation, marriage rating, gender)
# encode gender
X = pd.DataFrame({
    'children': children,
    'age': df['feature4'],
    'years_married': df['feature5'],
    'religiousness': df['feature7'],
    'education': df['feature8'],
    'occupation': df['feature9'],
    'marriage_rating': df['feature10'],
})
# gender binary male=1
X['male'] = df['feature3'].astype(str).str.lower().map({'male':1,'female':0})
X = sm.add_constant(X)
ols = sm.OLS(y, X, missing='drop').fit()

# Logit on any affair >0
any_affair = (y > 0).astype(int)
logit = sm.Logit(any_affair, X, missing='drop').fit(disp=False)

summary = {
    'group_stats': stats_by.to_dict(),
    't_test': {'stat': float(t_res.statistic), 'pvalue': float(t_res.pvalue)},
    'mannwhitney': {'stat': float(u_res.statistic), 'pvalue': float(u_res.pvalue)},
    'ols_children_coef': float(ols.params['children']),
    'ols_children_pvalue': float(ols.pvalues['children']),
    'logit_children_coef': float(logit.params['children']),
    'logit_children_pvalue': float(logit.pvalues['children']),
}

with open('analysis_output.json','w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
