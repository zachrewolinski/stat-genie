import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('mortgage.csv')

# Basic counts
n = len(df)

# Ensure binary columns
female = df['female']
accept = df['accept']

# Acceptance rates by gender
rates = df.groupby('female')['accept'].mean()
counts = df.groupby('female')['accept'].agg(['count','sum'])

# Chi-square test of independence (female vs accept)
contingency = pd.crosstab(df['female'], df['accept'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Unadjusted logistic regression: accept ~ female
X_unadj = sm.add_constant(df['female'])
model_unadj = sm.Logit(df['accept'], X_unadj, missing='drop')
res_unadj = model_unadj.fit(disp=0)

# Adjusted logistic regression with typical credit-related covariates
covariates = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
X_adj = sm.add_constant(df[covariates])
model_adj = sm.Logit(df['accept'], X_adj, missing='drop')
res_adj = model_adj.fit(disp=0)

# Extract key stats
female_unadj_coef = res_unadj.params['female']
female_unadj_p = res_unadj.pvalues['female']

female_adj_coef = res_adj.params['female']
female_adj_p = res_adj.pvalues['female']

# Odds ratios
odds_unadj = np.exp(female_unadj_coef)
odds_adj = np.exp(female_adj_coef)

print('n:', n)
print('acceptance rates by female (0=male,1=female):')
print(rates)
print('counts by female (count, accepted sum):')
print(counts)
print('chi2:', chi2, 'p:', p_chi)
print('unadjusted logit coef female:', female_unadj_coef, 'p:', female_unadj_p, 'odds ratio:', odds_unadj)
print('adjusted logit coef female:', female_adj_coef, 'p:', female_adj_p, 'odds ratio:', odds_adj)

# Also compute difference in rates and CI for difference in proportions
# Using normal approximation
rate_male = rates.loc[0.0] if 0.0 in rates.index else rates.loc[0]
rate_female = rates.loc[1.0] if 1.0 in rates.index else rates.loc[1]
count_male = counts.loc[0.0, 'count'] if 0.0 in counts.index else counts.loc[0, 'count']
count_female = counts.loc[1.0, 'count'] if 1.0 in counts.index else counts.loc[1, 'count']

p1 = rate_female
p0 = rate_male
se = np.sqrt(p1*(1-p1)/count_female + p0*(1-p0)/count_male)

diff = p1 - p0
z = diff / se if se > 0 else np.nan
p_two = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan

print('difference in acceptance rates (female - male):', diff)
print('z:', z, 'p:', p_two)
