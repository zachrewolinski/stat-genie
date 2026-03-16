import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Basic checks
# Ensure binary columns are numeric

# Acceptance rate by gender
summary = df.groupby('female')[['accept','deny']].mean()
count = df.groupby('female').size()

# Two-proportion z-test for acceptance rates (female vs male)
# Use accept rates
female_accept = df[df['female']==1]['accept']
male_accept = df[df['female']==0]['accept']

n1 = female_accept.shape[0]
n0 = male_accept.shape[0]

p1 = female_accept.mean()
p0 = male_accept.mean()

# pooled proportion
p_pool = (female_accept.sum() + male_accept.sum()) / (n1 + n0)

# z-test
import math
se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
if se > 0:
    z = (p1 - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = float('nan')
    p_value = float('nan')

# Chi-square test of independence
contingency = pd.crosstab(df['female'], df['accept'])
chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression: accept ~ female (unadjusted)
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=False)

# Logistic regression with controls
# Use available covariates; avoid multicollinearity with accept/deny
covariates = [
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]

# Drop rows with missing values
model_df = df[covariates + ['accept']].dropna()

formula = 'accept ~ ' + ' + '.join(covariates)
model_adj = smf.logit(formula, data=model_df).fit(disp=False)

# Output key results
print('Counts by female:', count.to_dict())
print('Acceptance rates by female:', summary['accept'].to_dict())
print('Two-proportion z-test: z=%.4f, p=%.6f' % (z, p_value))
print('Chi-square: chi2=%.4f, p=%.6f' % (chi2, chi2_p))

print('\nUnadjusted logistic regression:')
print(model_unadj.summary().tables[1])

print('\nAdjusted logistic regression:')
print(model_adj.summary().tables[1])

# Odds ratios for female
unadj_or = float(pd.Series(model_unadj.params).apply(lambda x: math.exp(x))['female'])
adj_or = float(pd.Series(model_adj.params).apply(lambda x: math.exp(x))['female'])

print('\nOdds ratio female (unadjusted):', unadj_or)
print('Odds ratio female (adjusted):', adj_or)
