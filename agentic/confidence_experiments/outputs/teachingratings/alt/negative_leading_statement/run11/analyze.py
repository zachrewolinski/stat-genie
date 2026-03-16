import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('teachingratings.csv')

# Basic checks
n = len(df)

# Correlation between beauty and eval
corr = df['beauty'].corr(df['eval'])

# Simple OLS
m1 = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# OLS with controls (common in literature)
# Use categorical variables with C(), include age, gender, minority, native, tenure, division, credits, class size
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students'

m2 = smf.ols(formula, data=df).fit(cov_type='HC3')

# Effect size per 1 SD of beauty
beauty_sd = df['beauty'].std(ddof=0)

m1_beta = m1.params['beauty']
m1_se = m1.bse['beauty']
m1_p = m1.pvalues['beauty']

m2_beta = m2.params['beauty']
m2_se = m2.bse['beauty']
m2_p = m2.pvalues['beauty']

# predicted change for 1 sd beauty
m1_change = m1_beta * beauty_sd
m2_change = m2_beta * beauty_sd

result = {
    'n': int(n),
    'corr_beauty_eval': float(corr),
    'm1': {
        'beta': float(m1_beta),
        'se': float(m1_se),
        'p': float(m1_p),
        'r2': float(m1.rsquared),
        'change_per_sd': float(m1_change),
    },
    'm2': {
        'beta': float(m2_beta),
        'se': float(m2_se),
        'p': float(m2_p),
        'r2': float(m2.rsquared),
        'change_per_sd': float(m2_change),
    },
    'beauty_sd': float(beauty_sd),
    'eval_mean': float(df['eval'].mean()),
    'eval_sd': float(df['eval'].std(ddof=0)),
}

print(json.dumps(result, indent=2))
