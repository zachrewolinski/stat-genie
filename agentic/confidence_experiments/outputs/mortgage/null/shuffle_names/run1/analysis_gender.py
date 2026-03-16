import pandas as pd
import statsmodels.api as sm
import numpy as np


df = pd.read_csv('mortgage.csv')

# gender column per metadata
female_col = 'denied_PMI'  # 1 female, 0 male per info.json

# candidate outcome columns
outcomes = {
    'self_employed': 'denial_indicator',
    'deny': 'acceptance_indicator',
    'accept': 'self_employed_indicator',
}

# controls (creditworthiness) based on metadata mapping
controls = ['mortgage_credit','housing_expense_ratio','Unnamed: 0','PI_ratio','loan_to_value','consumer_credit','married','black']
# remove controls not in df
controls = [c for c in controls if c in df.columns]

print('controls', controls)

for outcome in ['self_employed','deny']:
    data = df[[outcome, female_col] + controls].dropna()
    y = data[outcome]
    X = data[[female_col] + controls]
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X).fit(disp=False)
    print('\nOutcome', outcome)
    print('n', len(data))
    print('female coef', model.params[female_col], 'p', model.pvalues[female_col])
    # compute marginal difference in outcome rate by female
    rate_f = data[data[female_col]==1][outcome].mean()
    rate_m = data[data[female_col]==0][outcome].mean()
    print('rate female', rate_f, 'rate male', rate_m, 'diff', rate_f - rate_m)

# also chi-square test for difference in proportions
from statsmodels.stats.proportion import proportions_ztest

for outcome in ['self_employed','deny']:
    data = df[[outcome, female_col]].dropna()
    count = [data[data[female_col]==1][outcome].sum(), data[data[female_col]==0][outcome].sum()]
    nobs = [data[data[female_col]==1][outcome].count(), data[data[female_col]==0][outcome].count()]
    stat, pval = proportions_ztest(count, nobs)
    print('\nOutcome', outcome, 'ztest p', pval)
