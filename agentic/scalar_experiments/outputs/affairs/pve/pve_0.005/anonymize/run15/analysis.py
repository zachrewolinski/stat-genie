import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('affairs.csv')

# Ensure expected columns
required_cols = [
    'feature1','feature2','feature3','feature4','feature5',
    'feature6','feature7','feature8','feature9','feature10'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Clean/prepare
# Feature2 is affair frequency; feature6 is children yes/no

df['children'] = df['feature6'].map({'yes': 1, 'no': 0})

df['any_affair'] = (df['feature2'] > 0).astype(int)

# Group stats
summary = df.groupby('children')['feature2'].agg(['count','mean','median','std'])

# t-test on means (Welch)
no = df[df['children'] == 0]['feature2']
yes = df[df['children'] == 1]['feature2']

t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False)

# Mann-Whitney U test (nonparametric)
try:
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Cohen's d (yes - no)
mean_diff = yes.mean() - no.mean()
pooled_std = np.sqrt(((yes.std(ddof=1) ** 2) + (no.std(ddof=1) ** 2)) / 2)
cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

# OLS regression for frequency with controls
# Treat gender as categorical, children as indicator
formula = (
    'feature2 ~ children + C(feature3) + feature4 + feature5 + '
    'feature7 + feature8 + feature9 + feature10'
)
ols_model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Logistic regression for any affair
logit_formula = (
    'any_affair ~ children + C(feature3) + feature4 + feature5 + '
    'feature7 + feature8 + feature9 + feature10'
)
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

# Extract key results
ols_children_coef = ols_model.params['children']
ols_children_p = ols_model.pvalues['children']

logit_children_coef = logit_model.params['children']
logit_children_p = logit_model.pvalues['children']
logit_or = float(np.exp(logit_children_coef))

# Simple proportions for any_affair
prop_any = df.groupby('children')['any_affair'].mean()

results = {
    'group_summary': summary.to_dict(),
    'mean_diff_yes_minus_no': float(mean_diff),
    'cohens_d_yes_minus_no': float(cohens_d),
    't_test_p_value': float(t_p),
    't_stat': float(t_stat),
    'mannwhitney_p_value': float(u_p),
    'ols_children_coef': float(ols_children_coef),
    'ols_children_p': float(ols_children_p),
    'logit_children_or': float(logit_or),
    'logit_children_p': float(logit_children_p),
    'prop_any_affair_by_children': prop_any.to_dict(),
}

print(json.dumps(results, indent=2))
