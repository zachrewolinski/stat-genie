import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Map columns
# feature2: affairs frequency (0=none, higher codes)
# feature6: children yes/no

# Basic sanity
children = df['feature6'].astype(str)

affairs = df['feature2']

# Group stats
stats_by = df.groupby(children)['feature2'].agg(['count','mean','median'])

# Proportion with any affair (>0)
prop_any = df.assign(any_affair=affairs > 0).groupby(children)['any_affair'].mean()

# T-test of means
yes = affairs[children == 'yes']
no = affairs[children == 'no']

ttest = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney for non-normal
mwu = stats.mannwhitneyu(yes, no, alternative='two-sided')

# Regression controlling for covariates
# Use feature3 gender (categorical), feature4 age, feature5 years married, feature7 religiosity,
# feature8 education, feature9 occupation, feature10 marital happiness

X = df[['feature6','feature3','feature4','feature5','feature7','feature8','feature9','feature10']].copy()
X['feature6'] = (X['feature6'] == 'yes').astype(int)
X = pd.get_dummies(X, columns=['feature3'], drop_first=True)
X = sm.add_constant(X)

model = sm.OLS(affairs, X).fit()
coef_children = model.params['feature6']
pval_children = model.pvalues['feature6']

# Logistic regression for any affair
X_log = X.copy()
logit = sm.Logit((affairs > 0).astype(int), X_log).fit(disp=False)
coef_children_logit = logit.params['feature6']
pval_children_logit = logit.pvalues['feature6']

# Effect size (Cohen's d)
mean_yes = yes.mean()
mean_no = no.mean()
std_yes = yes.std(ddof=1)
std_no = no.std(ddof=1)

# Pooled SD for Cohen's d
n_yes = yes.shape[0]
n_no = no.shape[0]
pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

print('Group stats:\n', stats_by)
print('\nProp any affair:\n', prop_any)
print('\nT-test:', ttest)
print('Mann-Whitney:', mwu)
print('\nOLS coef children:', coef_children, 'p=', pval_children)
print('Logit coef children:', coef_children_logit, 'p=', pval_children_logit)
print('Cohen d:', cohens_d)
