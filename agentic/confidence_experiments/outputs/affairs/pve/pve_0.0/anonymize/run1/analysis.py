import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Key variables
outcome = 'feature2'  # extramarital affairs frequency
children = 'feature6'  # yes/no

# Basic checks
print('Columns:', df.columns.tolist())
print('Rows:', len(df))
print('Children value counts:')
print(df[children].value_counts(dropna=False))
print('\nOutcome summary:')
print(df[outcome].describe())

# Grouped stats
stats_by_child = df.groupby(children)[outcome].agg(['count','mean','median','std'])
stats_by_child['prop_any_affair'] = df.groupby(children)[outcome].apply(lambda x: (x>0).mean())
print('\nGroup stats by children:')
print(stats_by_child)

# Two-sample tests
no = df[df[children]=='no'][outcome]
yes = df[df[children]=='yes'][outcome]

# Welch t-test
welch = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')
print('\nWelch t-test (yes vs no):', welch)

# Mann-Whitney U test (two-sided)
try:
    mwu = stats.mannwhitneyu(yes, no, alternative='two-sided')
    print('Mann-Whitney U:', mwu)
except Exception as e:
    print('Mann-Whitney error:', e)

# Difference in proportion any affairs
any_yes = (yes>0).mean(); any_no = (no>0).mean()
print('\nProportion any affair: yes={:.3f} no={:.3f} diff={:.3f}'.format(any_yes, any_no, any_yes-any_no))

# Two-proportion z-test
count = np.array([(yes>0).sum(), (no>0).sum()])
nobs = np.array([yes.notna().sum(), no.notna().sum()])
stat, pval = sm.stats.proportions_ztest(count, nobs)
print('Two-proportion z-test:', stat, pval)

# Regression: OLS on affair frequency
# Encode children yes=1
df2 = df.copy()
df2['children_yes'] = (df2[children] == 'yes').astype(int)
df2['male'] = (df2['feature3'] == 'male').astype(int)

# OLS with covariates
ols_formula = 'feature2 ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + male'
ols_model = smf.ols(ols_formula, data=df2).fit(cov_type='HC3')
print('\nOLS (HC3) summary - children coef:')
print(ols_model.params['children_yes'], ols_model.bse['children_yes'], ols_model.pvalues['children_yes'])

# Negative binomial (count-like)
try:
    nb_model = smf.glm(ols_formula, data=df2, family=sm.families.NegativeBinomial()).fit()
    print('NegBin coeff children:', nb_model.params['children_yes'], nb_model.bse['children_yes'], nb_model.pvalues['children_yes'])
except Exception as e:
    print('NegBin error:', e)

# Logistic regression for any affair
df2['any_affair'] = (df2['feature2'] > 0).astype(int)
logit_formula = 'any_affair ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + male'
logit_model = smf.logit(logit_formula, data=df2).fit(disp=False)
print('Logit children coef:', logit_model.params['children_yes'], logit_model.bse['children_yes'], logit_model.pvalues['children_yes'])

# Compute odds ratio
or_children = np.exp(logit_model.params['children_yes'])
print('Logit OR children:', or_children)
