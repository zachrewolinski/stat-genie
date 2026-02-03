import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.formula.api as smf

# Load data
# feature2: frequency of extramarital affairs
# feature6: children in marriage (yes/no)
# feature3: gender
# feature4: age
# feature5: years married
# feature7: religiosity
# feature8: education
# feature9: occupation
# feature10: marriage rating

df = pd.read_csv('affairs.csv')

# Basic group summaries
summary = df.groupby('feature6')['feature2'].agg(['count', 'mean', 'median'])
summary['any_affair_rate'] = df.groupby('feature6')['feature2'].apply(lambda s: (s > 0).mean())

# Two-sample t-test for mean affair frequency
with_children = df[df['feature6'] == 'yes']['feature2']
without_children = df[df['feature6'] == 'no']['feature2']

ttest = stats.ttest_ind(with_children, without_children, equal_var=False)

# Proportion test for any affair
any_with = (with_children > 0).sum()
any_without = (without_children > 0).sum()
counts = [any_with, any_without]
ns = [len(with_children), len(without_children)]
prop_test = proportions_ztest(counts, ns)

# Regression analyses controlling for covariates
# OLS on affair frequency
ols = smf.ols(
    'feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit()

# Logistic regression on any affair
# Create binary outcome

df['any_affair'] = (df['feature2'] > 0).astype(int)
logit = smf.logit(
    'any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit(disp=False)

# Print key results for inspection
print('Group summary by children (feature6):')
print(summary)
print('\nT-test (mean affair frequency, yes vs no children):')
print(ttest)
print('\nProportion test (any affair, yes vs no children):')
print('counts', counts, 'ns', ns, 'z, p', prop_test)

print('\nOLS coefficient for children (yes vs no):')
print(ols.params.get('C(feature6)[T.yes]'), ols.pvalues.get('C(feature6)[T.yes]'))

print('\nLogit coefficient for children (yes vs no):')
print(logit.params.get('C(feature6)[T.yes]'), logit.pvalues.get('C(feature6)[T.yes]'))
