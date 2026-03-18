import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')
print(_df.head())
print(_df.describe(include='all'))
print(_df['feature6'].value_counts())

# Group stats
grp = _df.groupby('feature6')['feature2']
print('group means', grp.mean())
print('group medians', grp.median())
print('group std', grp.std())

# t-test (unequal var)
yes = _df.loc[_df['feature6']=='yes','feature2']
no = _df.loc[_df['feature6']=='no','feature2']
print('n yes', len(yes), 'n no', len(no))

t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')
print('t-test', t_stat, p_val)

# Mann-Whitney
u_stat, p_u = stats.mannwhitneyu(yes, no, alternative='two-sided')
print('mannwhitney', u_stat, p_u)

# OLS with children only
m1 = smf.ols('feature2 ~ C(feature6)', data=_df).fit()
print(m1.summary())

# OLS with controls
m2 = smf.ols('feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=_df).fit()
print(m2.summary())

# effect size Cohen's d
mean_yes = yes.mean(); mean_no = no.mean()
var_yes = yes.var(ddof=1); var_no = no.var(ddof=1)
# pooled SD for unequal sizes
n1=len(yes); n2=len(no)
pooled_sd = np.sqrt(((n1-1)*var_yes + (n2-1)*var_no)/(n1+n2-2))
cohen_d = (mean_yes - mean_no)/pooled_sd
print('cohen d', cohen_d)
