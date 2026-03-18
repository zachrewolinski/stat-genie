import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic checks
print('columns', df.columns.tolist())
print('dtypes', df.dtypes)
print('head')
print(df.head())

# Identify variables
# feature6: children yes/no
# feature2: affairs frequency

# Clean
# Ensure feature6 categorical with yes/no

df['feature6'] = df['feature6'].astype(str).str.lower()

# Group summary
summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])
print('\nGroup summary for feature2 by children')
print(summary)

# Difference in means t-test (Welch)
children_yes = df.loc[df['feature6']=='yes','feature2']
children_no = df.loc[df['feature6']=='no','feature2']

# t-test
t_stat, p_val = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')
print('\nWelch t-test')
print('t', t_stat, 'p', p_val)

# Mann-Whitney U test (nonparametric)
try:
    u_stat, p_u = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
    print('\nMann-Whitney U')
    print('u', u_stat, 'p', p_u)
except Exception as e:
    print('Mann-Whitney failed', e)

# Effect size (Cohen d)
mean_diff = children_yes.mean() - children_no.mean()
# pooled SD for unequal sizes
n1, n2 = len(children_yes), len(children_no)
var1, var2 = children_yes.var(ddof=1), children_no.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
cohen_d = mean_diff / pooled_sd if pooled_sd>0 else np.nan
print('\nMean diff', mean_diff)
print('Cohen d', cohen_d)

# OLS regression with children as predictor
# encode children yes=1 no=0
df['children_yes'] = (df['feature6']=='yes').astype(int)

# Simple OLS
ols = smf.ols('feature2 ~ children_yes', data=df).fit()
print('\nOLS summary')
print(ols.summary())

# Optional: control for age, years married, religiosity, education, occupation, marital rating, gender
# Using feature3 as categorical (gender)

controls = ['feature4','feature5','feature7','feature8','feature9','feature10']
# feature3 categorical
ols2 = smf.ols('feature2 ~ children_yes + feature3 + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit()
print('\nOLS with controls')
print(ols2.summary())

# Save key results to json for reference
results = {
    'summary': summary.to_dict(),
    't_test': {'t': t_stat, 'p': p_val},
    'mannwhitney': {'u': float(u_stat), 'p': float(p_u)},
    'mean_diff': float(mean_diff),
    'cohen_d': float(cohen_d),
    'ols_coef': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
    'ols2_coef': ols2.params.to_dict(),
    'ols2_pvalues': ols2.pvalues.to_dict(),
}
with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)
