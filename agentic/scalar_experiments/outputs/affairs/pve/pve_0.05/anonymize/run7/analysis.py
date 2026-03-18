import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'affairs.csv'
df = pd.read_csv(csv_path)

# Map columns based on info.json
# feature2: frequency of affairs
# feature6: children yes/no

df = df.copy()

# Basic cleaning
df['has_children'] = df['feature6'].astype(str).str.lower().map({'yes':1, 'no':0})
# If mapping produced NaN for unexpected values, keep them

# outcome variables

df['affair_freq'] = pd.to_numeric(df['feature2'], errors='coerce')
df['affair_any'] = (df['affair_freq'] > 0).astype(int)

# Drop rows with missing critical values
base = df.dropna(subset=['affair_freq', 'has_children'])

# Descriptive stats
summary = base.groupby('has_children')['affair_freq'].agg(['count','mean','median','std'])
summary_any = base.groupby('has_children')['affair_any'].agg(['mean','count'])

# Two-sample t-test (unequal var) for affair frequency
children_yes = base.loc[base['has_children']==1, 'affair_freq']
children_no = base.loc[base['has_children']==0, 'affair_freq']

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparametric) for affair frequency
try:
    mwu = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
except Exception as e:
    mwu = None

# Logistic regression for any affair >0 (unadjusted)
logit_unadj = smf.logit('affair_any ~ has_children', data=base).fit(disp=False)

# Adjusted logistic regression with key covariates
# feature4 age, feature5 years married, feature7 religiosity, feature8 education, feature9 occupation, feature10 marriage rating, feature3 gender
# Convert gender to binary (female=1) to include
base['female'] = (base['feature3'].astype(str).str.lower()=='female').astype(int)

# Ensure numeric for covariates
covars = ['feature4','feature5','feature7','feature8','feature9','feature10','female']
for c in covars:
    base[c] = pd.to_numeric(base[c], errors='coerce')

adj = base.dropna(subset=covars)
logit_adj = smf.logit('affair_any ~ has_children + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + female', data=adj).fit(disp=False)

# OLS on affair frequency (unadjusted and adjusted) for interpretability
ols_unadj = smf.ols('affair_freq ~ has_children', data=base).fit()
ols_adj = smf.ols('affair_freq ~ has_children + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + female', data=adj).fit()

# Collect key results
results = {
    'summary_freq': summary,
    'summary_any': summary_any,
    'ttest_stat': ttest.statistic,
    'ttest_p': ttest.pvalue,
    'mwu_stat': None if mwu is None else mwu.statistic,
    'mwu_p': None if mwu is None else mwu.pvalue,
    'logit_unadj_coef': logit_unadj.params['has_children'],
    'logit_unadj_p': logit_unadj.pvalues['has_children'],
    'logit_unadj_or': np.exp(logit_unadj.params['has_children']),
    'logit_adj_coef': logit_adj.params['has_children'],
    'logit_adj_p': logit_adj.pvalues['has_children'],
    'logit_adj_or': np.exp(logit_adj.params['has_children']),
    'ols_unadj_coef': ols_unadj.params['has_children'],
    'ols_unadj_p': ols_unadj.pvalues['has_children'],
    'ols_adj_coef': ols_adj.params['has_children'],
    'ols_adj_p': ols_adj.pvalues['has_children'],
    'n_base': len(base),
    'n_adj': len(adj),
}

# Output results to console
print('Summary (affair_freq by children):')
print(summary)
print('\nSummary (affair_any by children):')
print(summary_any)
print('\nT-test:', ttest)
print('\nMWU:', mwu)
print('\nLogit unadj coef, p, OR:', results['logit_unadj_coef'], results['logit_unadj_p'], results['logit_unadj_or'])
print('Logit adj coef, p, OR:', results['logit_adj_coef'], results['logit_adj_p'], results['logit_adj_or'])
print('OLS unadj coef, p:', results['ols_unadj_coef'], results['ols_unadj_p'])
print('OLS adj coef, p:', results['ols_adj_coef'], results['ols_adj_p'])
print('n_base', results['n_base'], 'n_adj', results['n_adj'])

# Save results to a CSV for reference
pd.DataFrame({
    'metric': list(results.keys()),
    'value': list(results.values())
}).to_csv('analysis_results.csv', index=False)
