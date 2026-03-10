import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Map variables based on metadata descriptions:
# - Column `denied_PMI` described as "1 if applicant is female" -> gender
# - Column `deny` described as "1 if mortgage application was accepted" -> approval
# - Column `self_employed` described as "1 if mortgage application was denied" -> complement of approval

female_col = 'denied_PMI'
approve_col = 'deny'

# Basic rates
rates = df.groupby(female_col)[approve_col].mean()
counts = df.groupby(female_col)[approve_col].agg(['mean','count','sum'])
print('Approval rates by gender (female=1):')
print(counts)

# Difference in proportions (female - male)
# female=1, male=0
female_mask = df[female_col] == 1
male_mask = df[female_col] == 0

female_approve = df.loc[female_mask, approve_col]
male_approve = df.loc[male_mask, approve_col]

n_female = female_approve.shape[0]
n_male = male_approve.shape[0]

p_female = female_approve.mean()
p_male = male_approve.mean()

print('\nCounts: female', n_female, 'male', n_male)
print('Approval rate female', p_female, 'male', p_male, 'diff (female-male)', p_female - p_male)

# Two-proportion z-test
count = np.array([female_approve.sum(), male_approve.sum()])
nobs = np.array([n_female, n_male])
stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
print('Two-proportion z-test: z', stat, 'p', pval)

# Chi-square test for independence
ct = pd.crosstab(df[female_col], df[approve_col])
chi2, chi2_p, dof, expected = stats.chi2_contingency(ct)
print('Chi-square: chi2', chi2, 'p', chi2_p)

# Logistic regression
m1 = smf.logit(f"{approve_col} ~ {female_col}", data=df).fit(disp=False)
print('\nLogit unadjusted')
print(m1.params)
print(m1.pvalues)

# Adjusted model with controls (exclude outcome, its complement, ID)
controls = [
    'consumer_credit',
    'mortgage_credit',
    'accept',
    'loan_to_value',
    'married',
    'black',
    'PI_ratio',
    'housing_expense_ratio',
    'female',  # PMI denial per metadata
    'Unnamed: 0',  # loan-to-value per metadata
]

formula = f"{approve_col} ~ {female_col} + " + " + ".join(controls)

m2 = smf.logit(formula, data=df).fit(disp=False)
print('\nLogit adjusted (female effect)')
print('coef', m2.params[female_col], 'p', m2.pvalues[female_col])

# Odds ratio for female in adjusted model
odds_ratio = np.exp(m2.params[female_col])
print('Adjusted odds ratio (female):', odds_ratio)

