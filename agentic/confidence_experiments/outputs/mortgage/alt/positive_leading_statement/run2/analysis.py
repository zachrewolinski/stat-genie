import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')

# Basic counts
counts = df['female'].value_counts().sort_index()

# Denial rates by gender
rates = df.groupby('female')['deny'].mean()

# Two-proportion z-test for denial rates
# female=1 vs male=0
count = np.array([
    df.loc[df['female'] == 1, 'deny'].sum(),
    df.loc[df['female'] == 0, 'deny'].sum(),
])
obs = np.array([
    (df['female'] == 1).sum(),
    (df['female'] == 0).sum(),
])
stat, pval = proportions_ztest(count, obs)

# Logistic regression with controls
# Exclude accept (complement) and index column
predictors = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

formula = 'deny ~ ' + ' + '.join(predictors)
model = smf.logit(formula=formula, data=df).fit(disp=False)

female_coef = model.params['female']
female_se = model.bse['female']
female_p = model.pvalues['female']

# Odds ratio and 95% CI
odds_ratio = float(np.exp(female_coef))
ci_low, ci_high = np.exp(model.conf_int().loc['female'])

# Print summary stats
print('N', len(df))
print('Counts female', counts.to_dict())
print('Denial rates by female', rates.to_dict())
print('Z-test stat', stat, 'p', pval)
print('Logit female coef', female_coef, 'se', female_se, 'p', female_p)
print('Odds ratio', odds_ratio, 'CI', (float(ci_low), float(ci_high)))
print(model.summary())
