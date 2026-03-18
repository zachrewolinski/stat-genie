import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
file_path = 'affairs.csv'
df = pd.read_csv(file_path)

# Map children to binary
if df['feature6'].dtype == object:
    df['children'] = df['feature6'].str.strip().str.lower().map({'yes':1, 'no':0})
else:
    df['children'] = df['feature6']

# Outcomes
outcome = df['feature2']

# Descriptive stats by children
summary = df.groupby('children')['feature2'].agg(['count','mean','median','std'])

# Welch t-test on mean difference
with_children = df.loc[df['children']==1, 'feature2']
without_children = df.loc[df['children']==0, 'feature2']

welch_t = stats.ttest_ind(with_children, without_children, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparam)
try:
    mwu = stats.mannwhitneyu(with_children, without_children, alternative='two-sided')
except Exception:
    mwu = None

# Effect size (Cohen's d for Welch)
mean_diff = with_children.mean() - without_children.mean()
# pooled SD (approx) for d
n1 = with_children.dropna().shape[0]
n0 = without_children.dropna().shape[0]
var1 = with_children.var(ddof=1)
var0 = without_children.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohens_d = mean_diff / pooled_sd

# Probability of any affair (>0)
df['any_affair'] = (df['feature2'] > 0).astype(int)
any_summary = df.groupby('children')['any_affair'].mean()

# Logistic regression for any affair ~ children
logit_model = smf.logit('any_affair ~ children', data=df).fit(disp=0)

# OLS regression for frequency (continuous)
ols_model = smf.ols('feature2 ~ children', data=df).fit()

# Adjusted models controlling for likely confounders
# using available features: age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10), gender (feature3)
# Handle gender as categorical
adj_formula = 'feature2 ~ children + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)'
adj_ols = smf.ols(adj_formula, data=df).fit()

adj_logit = smf.logit('any_affair ~ children + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)', data=df).fit(disp=0)

results = {
    'summary': summary,
    'welch_t': welch_t,
    'mwu': mwu,
    'mean_diff': mean_diff,
    'cohens_d': cohens_d,
    'any_summary': any_summary,
    'logit_coef': logit_model.params,
    'logit_pvalues': logit_model.pvalues,
    'ols_coef': ols_model.params,
    'ols_pvalues': ols_model.pvalues,
    'adj_ols_coef': adj_ols.params,
    'adj_ols_pvalues': adj_ols.pvalues,
    'adj_logit_coef': adj_logit.params,
    'adj_logit_pvalues': adj_logit.pvalues,
}

# Print key outputs
print('Summary by children:')
print(summary)
print('\nMean diff (children - no children):', mean_diff)
print('Cohen d:', cohens_d)
print('\nWelch t-test:', welch_t)
print('\nMann-Whitney:', mwu)
print('\nAny affair rate by children:')
print(any_summary)
print('\nLogit (any affair ~ children):')
print('coef:', logit_model.params['children'], 'p:', logit_model.pvalues['children'])
print('\nOLS (feature2 ~ children):')
print('coef:', ols_model.params['children'], 'p:', ols_model.pvalues['children'])
print('\nAdjusted OLS (feature2 ~ children + controls):')
print('coef:', adj_ols.params['children'], 'p:', adj_ols.pvalues['children'])
print('\nAdjusted Logit (any affair ~ children + controls):')
print('coef:', adj_logit.params['children'], 'p:', adj_logit.pvalues['children'])
