import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('affairs.csv')

# According to metadata, column 'religiousness' indicates whether there are children in the marriage.
children = df['religiousness'].map({'yes': 1, 'no': 0})
# According to metadata, column 'age' represents frequency of extramarital affairs.
affairs = df['age']

# Drop missing (just in case)
mask = children.notna() & affairs.notna()
children = children[mask]
affairs = affairs[mask]

# Group stats
mean_with = affairs[children == 1].mean()
mean_without = affairs[children == 0].mean()
std_with = affairs[children == 1].std(ddof=1)
std_without = affairs[children == 0].std(ddof=1)

n_with = (children == 1).sum()
n_without = (children == 0).sum()

# Welch t-test
t_stat, p_val = stats.ttest_ind(affairs[children==1], affairs[children==0], equal_var=False)

# Cohen's d (using pooled SD)
pooled_sd = np.sqrt(((n_with-1)*std_with**2 + (n_without-1)*std_without**2) / (n_with+n_without-2))
cohen_d = (mean_with - mean_without) / pooled_sd

# OLS regression with robust SE
X = sm.add_constant(children)
ols = sm.OLS(affairs, X).fit(cov_type='HC3')

print('n_with', n_with, 'n_without', n_without)
print('mean_with', mean_with, 'mean_without', mean_without)
print('std_with', std_with, 'std_without', std_without)
print('t_stat', t_stat, 'p_val', p_val)
print('cohen_d', cohen_d)
print('ols params', ols.params)
print('ols pvalues', ols.pvalues)
print('ols confint', ols.conf_int())
