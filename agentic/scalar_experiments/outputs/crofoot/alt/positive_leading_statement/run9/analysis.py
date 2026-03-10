import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived predictors
df['rel_size'] = df['n_focal'] - df['n_other']
# positive means contest closer to focal group's home range center relative to other
df['rel_dist_100'] = (df['dist_other'] - df['dist_focal']) / 100.0

# Logistic regression models
model_full = smf.glm('win ~ rel_size + rel_dist_100', data=df, family=sm.families.Binomial()).fit()
model_size = smf.glm('win ~ rel_size', data=df, family=sm.families.Binomial()).fit()
model_dist = smf.glm('win ~ rel_dist_100', data=df, family=sm.families.Binomial()).fit()

# Odds ratios and CI
params = model_full.params
conf = model_full.conf_int()
conf.columns = ['2.5%', '97.5%']

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

# Summaries
print('N:', len(df))
print('\nModel: win ~ rel_size + rel_dist_100')
print(model_full.summary())

print('\nOdds Ratios (per unit):')
print(pd.DataFrame({
    'OR': odds_ratios,
    'CI_low': conf_or['2.5%'],
    'CI_high': conf_or['97.5%'],
    'p_value': model_full.pvalues
}))

print('\nModel: win ~ rel_size')
print(model_size.summary())

print('\nModel: win ~ rel_dist_100')
print(model_dist.summary())

# Basic descriptive checks
print('\nDescriptives:')
print(df[['win', 'rel_size', 'rel_dist_100']].describe())

# Win rates by rel_size sign and rel_dist sign
for col in ['rel_size', 'rel_dist_100']:
    sign = np.where(df[col] > 0, 'positive', np.where(df[col] < 0, 'negative', 'zero'))
    tmp = df.groupby(sign)['win'].agg(['mean', 'count'])
    print(f"\nWin rate by {col} sign:")
    print(tmp)
