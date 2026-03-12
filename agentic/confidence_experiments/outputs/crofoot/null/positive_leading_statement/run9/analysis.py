import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Derived variables

df['rel_size'] = df['n_focal'] - df['n_other']
# Positive rel_dist means other group is farther from its home-range center than focal
# (i.e., contest is closer to focal group's center).
df['rel_dist'] = df['dist_other'] - df['dist_focal']

# Summary
print('rows', len(df))
print(df[['win', 'rel_size', 'rel_dist']].describe())

# Logistic regression: win ~ rel_size + rel_dist
X = df[['rel_size', 'rel_dist']]
X = sm.add_constant(X)
model = sm.GLM(df['win'], X, family=sm.families.Binomial())
res = model.fit()
print(res.summary())

# Univariate models
for col in ['rel_size', 'rel_dist']:
    X1 = sm.add_constant(df[[col]])
    m1 = sm.GLM(df['win'], X1, family=sm.families.Binomial())
    r1 = m1.fit()
    print('\nUnivariate', col)
    print(r1.summary())

# Odds ratios and CI
params = res.params
conf = res.conf_int()
or_df = pd.DataFrame(
    {
        'coef': params,
        'OR': np.exp(params),
        'CI_low': np.exp(conf[0]),
        'CI_high': np.exp(conf[1]),
        'p': res.pvalues,
    }
)
print('\nOdds ratios')
print(or_df)
