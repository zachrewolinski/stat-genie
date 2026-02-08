import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('affairs.csv')

# Basic group comparisons
any_aff = (_df['affairs'] > 0).astype(int)
means = _df.groupby('children')['affairs'].mean()
rates = _df.assign(any_aff=any_aff).groupby('children')['any_aff'].mean()

# Welch t-test on affairs counts
no = _df[_df['children'] == 'no']['affairs']
yes = _df[_df['children'] == 'yes']['affairs']
_ttest = stats.ttest_ind(yes, no, equal_var=False)

# Logit on any affairs
X = pd.get_dummies(_df[['children']], drop_first=True)
X = sm.add_constant(X)
logit = sm.Logit(any_aff, X).fit(disp=0)

# Poisson on affairs counts
poisson = sm.GLM(_df['affairs'], X, family=sm.families.Poisson()).fit()

print('mean affairs by children:\n', means)
print('any affairs rate by children:\n', rates)
print('welch t-test:', _ttest)
print('logit params/pvalues:', logit.params.to_dict(), logit.pvalues.to_dict())
print('poisson params/pvalues:', poisson.params.to_dict(), poisson.pvalues.to_dict())
