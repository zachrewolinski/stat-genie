import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = pd.read_csv('crofoot.csv')

# basic checks
print('rows', len(df))
print(df.head())

# Define relative group size and relative contest location
# Relative group size: focal size - other size
# Relative location: focal distance from center minus other distance from center
# Positive rel_dist means contest closer to focal group's center (focal is closer than other)

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_other'] - df['dist_focal']  # positive if contest closer to focal than other? check

# Let's compute both options for clarity

# outcome
Y = df['win']

# logistic regression with rel_size and rel_dist
X = df[['rel_size', 'rel_dist']].astype(float)
X = sm.add_constant(X)
model = sm.Logit(Y, X).fit(disp=False)
print(model.summary())

# odds ratios and 95% CI
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
OR = np.exp(params)
OR_ci = np.exp(conf)

results = pd.DataFrame({'coef': params, 'OR': OR, 'OR_2.5%': OR_ci['2.5%'], 'OR_97.5%': OR_ci['97.5%'], 'p': model.pvalues})
print(results)

# Alternative relative location sign: focal distance - other distance
# if we define rel_dist2 = dist_focal - dist_other

df['rel_dist2'] = df['dist_focal'] - df['dist_other']
X2 = df[['rel_size', 'rel_dist2']].astype(float)
X2 = sm.add_constant(X2)
model2 = sm.Logit(Y, X2).fit(disp=False)
print(model2.summary())

params2 = model2.params
conf2 = model2.conf_int()
conf2.columns = ['2.5%', '97.5%']
OR2 = np.exp(params2)
OR2_ci = np.exp(conf2)
results2 = pd.DataFrame({'coef': params2, 'OR': OR2, 'OR_2.5%': OR2_ci['2.5%'], 'OR_97.5%': OR2_ci['97.5%'], 'p': model2.pvalues})
print(results2)

# Univariate models to see effect
for var in ['rel_size', 'rel_dist', 'rel_dist2', 'n_focal', 'n_other', 'dist_focal', 'dist_other']:
    Xv = sm.add_constant(df[[var]].astype(float))
    m = sm.Logit(Y, Xv).fit(disp=False)
    print('\n', var)
    print(m.summary())

