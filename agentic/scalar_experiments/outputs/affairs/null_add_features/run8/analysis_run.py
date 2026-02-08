import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

path = 'affairs.csv'
df = pd.read_csv(path)

# Basic metrics

df['has_affair'] = (df['affairs'] > 0).astype(int)

summary = df.groupby('children').agg(
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    prop_affair=('has_affair','mean'),
    n=('affairs','size')
)
print('summary by children')
print(summary)

# t-test for mean affairs
yes = df[df['children']=='yes']['affairs']
no = df[df['children']=='no']['affairs']

# Welch t-test
w_t, w_p = stats.ttest_ind(yes, no, equal_var=False)
print('welch_t', w_t, 'p', w_p)

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
    print('mannwhitney_u', u_stat, 'p', u_p)
except Exception as e:
    print('mannwhitney error', e)

# Logistic regression for any affair controlling for covariates
# Use relevant predictors only
covariates = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
# Gender as binary
# children yes=1

df_model = df.copy()

df_model['children_yes'] = (df_model['children']=='yes').astype(int)

df_model['gender_male'] = (df_model['gender']=='male').astype(int)

X = df_model[['children_yes','gender_male'] + covariates]
X = sm.add_constant(X)
y = df_model['has_affair']

logit = sm.Logit(y, X).fit(disp=False)
print('logit params')
print(logit.params)
print('logit pvalues')
print(logit.pvalues)

# OLS on log1p(affairs)

df_model['log_affairs'] = np.log1p(df_model['affairs'])

ols = sm.OLS(df_model['log_affairs'], X).fit()
print('ols params')
print(ols.params)
print('ols pvalues')
print(ols.pvalues)

# Effect size for proportion difference (risk difference)
prop_yes = summary.loc['yes','prop_affair']
prop_no = summary.loc['no','prop_affair']
print('prop_yes', prop_yes, 'prop_no', prop_no, 'diff yes-no', prop_yes - prop_no)

# Cohen d for affairs
# Use pooled SD
n1, n2 = len(yes), len(no)
var1, var2 = yes.var(ddof=1), no.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
cohen_d = (yes.mean() - no.mean()) / pooled_sd
print('cohen_d (yes-no)', cohen_d)
