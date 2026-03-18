import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Basic cleaning
# Feature mapping based on info.json
# feature2: affair frequency (continuous)
# feature6: children yes/no

# Drop rows with missing values in key fields
key_cols = ['feature2', 'feature6']
sub = df.dropna(subset=key_cols).copy()

# Indicator: has_children
sub['has_children'] = sub['feature6'].map({'yes': 1, 'no': 0})

# Group stats
grp = sub.groupby('has_children')['feature2']
summary = grp.agg(['count', 'mean', 'median', 'std']).reset_index()

# Effect size: Cohen's d (using pooled SD)
vals_yes = sub.loc[sub['has_children'] == 1, 'feature2']
vals_no = sub.loc[sub['has_children'] == 0, 'feature2']

n1, n0 = len(vals_yes), len(vals_no)
mean1, mean0 = vals_yes.mean(), vals_no.mean()
std1, std0 = vals_yes.std(ddof=1), vals_no.std(ddof=1)

# pooled SD
pooled_sd = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2) / (n1 + n0 - 2))
cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

# Welch's t-test
welch_t = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)
try:
    mwu = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
except Exception as e:
    mwu = e

# OLS regression with controls
# Controls: feature3 gender, feature4 age, feature5 years married, feature7 relig, feature8 education, feature9 occupation, feature10 marital rating
# Use robust SE (HC3)
formula = 'feature2 ~ has_children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
model = smf.ols(formula, data=sub).fit(cov_type='HC3')

# Unadjusted OLS
a_model = smf.ols('feature2 ~ has_children', data=sub).fit(cov_type='HC3')

print('Summary by children (0=no,1=yes):')
print(summary.to_string(index=False))
print('\nMean difference (yes - no):', mean1 - mean0)
print('Cohen d:', cohens_d)
print('Welch t-test:', welch_t)
print('Mann-Whitney U:', mwu)

print('\nUnadjusted regression (feature2 ~ has_children):')
print(a_model.summary().tables[1])

print('\nAdjusted regression:')
print(model.summary().tables[1])

# Also compute proportion with zero affairs? Maybe compare probability of zero engagement.
# Define zero as <=0? Since noise may make negative. We'll use <=0 threshold.
sub['affair_zero'] = (sub['feature2'] <= 0).astype(int)
zero_rate = sub.groupby('has_children')['affair_zero'].mean().reset_index()
print('\nZero (<=0) rate by children:')
print(zero_rate.to_string(index=False))

# Logistic regression for zero vs nonzero
logit_model = smf.logit('affair_zero ~ has_children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=sub).fit(disp=False)
print('\nAdjusted logit for zero affairs:')
print(logit_model.summary().tables[1])
