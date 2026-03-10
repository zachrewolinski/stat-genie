import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create predictors
# Relative group size (focal - other)
df['size_diff'] = df['n_focal'] - df['n_other']
# Location advantage: positive if focal is closer to its home center than other is to its home center
# (dist_other - dist_focal)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize for effect size comparison (optional)
# We'll keep raw for interpretability

# Fit logistic regression
model = logit('win ~ size_diff + loc_adv', data=df).fit(disp=0)

# Also univariate models
model_size = logit('win ~ size_diff', data=df).fit(disp=0)
model_loc = logit('win ~ loc_adv', data=df).fit(disp=0)

# Summaries
print('N:', len(df))
print('win rate:', df['win'].mean())
print('\nMain model summary:')
print(model.summary())

# Odds ratios
params = model.params
conf = model.conf_int()
ors = np.exp(params)
conf_or = np.exp(conf)
print('\nOdds ratios (main model):')
for name in params.index:
    print(name, ors[name], conf_or.loc[name, 0], conf_or.loc[name, 1], model.pvalues[name])

print('\nUnivariate size model:')
print(model_size.summary())
print('OR size:', np.exp(model_size.params['size_diff']), 'p:', model_size.pvalues['size_diff'])

print('\nUnivariate location model:')
print(model_loc.summary())
print('OR loc_adv:', np.exp(model_loc.params['loc_adv']), 'p:', model_loc.pvalues['loc_adv'])

# Correlation between predictors
print('\nCorrelation size_diff vs loc_adv:', df['size_diff'].corr(df['loc_adv']))

# For effect interpretation: compute predicted probabilities for representative values
# Use mean loc_adv and size_diff
mean_loc = df['loc_adv'].mean()
mean_size = df['size_diff'].mean()

# Size effect: -5 to +5 around mean
sizes = np.array([-5, -2, 0, 2, 5])
locs = np.array([-200, -100, 0, 100, 200])

print('\nPredicted win probs varying size_diff (loc at mean):')
for s in sizes:
    X = pd.DataFrame({'size_diff':[s], 'loc_adv':[mean_loc]})
    p = model.predict(X)[0]
    print(s, p)

print('\nPredicted win probs varying loc_adv (size at mean):')
for l in locs:
    X = pd.DataFrame({'size_diff':[mean_size], 'loc_adv':[l]})
    p = model.predict(X)[0]
    print(l, p)
