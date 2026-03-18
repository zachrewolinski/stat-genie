import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Identify columns
cols = df.columns.tolist()
print('columns', cols)

# Map known columns
# feature2: affair frequency
# feature6: children yes/no

# Clean
# Ensure feature6 is categorical yes/no

df = df.copy()

# Remove missing

df = df.dropna(subset=['feature2','feature6'])

# Basic group stats

# convert to numeric

df['feature2'] = pd.to_numeric(df['feature2'])

# binary any affair

df['any_affair'] = (df['feature2'] > 0).astype(int)

# group

groups = df.groupby('feature6')

summary = groups['feature2'].agg(['count','mean','median','std'])

# proportion any affair

prop_any = groups['any_affair'].mean()

print('summary', summary)
print('prop_any', prop_any)

# Welch t-test on feature2

yes = df[df['feature6']=='yes']['feature2']
no = df[df['feature6']=='no']['feature2']

# t-test

t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False)

# Mann-Whitney U

u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')

# Cohen's d (using pooled SD)

n1, n2 = len(yes), len(no)

s1, s2 = yes.std(ddof=1), no.std(ddof=1)

s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))

cohen_d = (yes.mean() - no.mean())/s_pooled

print('t', t_stat, t_p)
print('u', u_stat, u_p)
print('cohen_d', cohen_d)

# Difference in proportions any affair (chi-square)
cont = pd.crosstab(df['feature6'], df['any_affair'])

chi2, chi_p, dof, expected = stats.chi2_contingency(cont)

print('contingency', cont)
print('chi2', chi2, chi_p)

# Logistic regression for any_affair with controls
# Use available numeric controls; treat feature3 gender and feature6 children as categorical

# Build formula
# Include age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10), gender (feature3)

formula = 'any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'

logit = smf.logit(formula, data=df).fit(disp=False)

print(logit.summary())

# OLS on log1p affair frequency
# Some noisy values may be negative; clip at 0 for log transform.
df['log_affair'] = np.log1p(np.clip(df['feature2'], 0, None))

ols = smf.ols('log_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(cov_type='HC3')

print(ols.summary())

# Save key metrics

results = {
    'summary': summary.to_dict(),
    'prop_any': prop_any.to_dict(),
    't_stat': float(t_stat),
    't_p': float(t_p),
    'u_stat': float(u_stat),
    'u_p': float(u_p),
    'cohen_d': float(cohen_d),
    'chi2': float(chi2),
    'chi_p': float(chi_p),
    'logit_params': logit.params.to_dict(),
    'logit_pvalues': logit.pvalues.to_dict(),
    'ols_params': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
    'n': int(len(df))
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('saved analysis_results.json')
