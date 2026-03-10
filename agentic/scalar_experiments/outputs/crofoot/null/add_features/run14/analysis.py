import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'crofoot.csv'
df = pd.read_csv(path)

# compute relative size and relative location
# relative size: focal - other
# relative location: contest is closer to focal center if dist_focal < dist_other
# Use difference (other - focal) so positive means closer to focal? let's do dist_other - dist_focal
# Actually if dist_focal is smaller, contest is closer to focal. So define rel_loc = dist_other - dist_focal


df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_loc'] = df['dist_other'] - df['dist_focal']

# logistic regression
X = df[['rel_size','rel_loc']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X).fit(disp=False)
print(model.summary())

# also compute odds ratios and CIs
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

print('\nOdds ratios:')
print(odds)
print('\n95% CI (odds):')
print(conf_odds)

# simple bivariate correlations
print('\nWin rate by rel_size:')
print(df.groupby('rel_size')['win'].mean())

# check model with standardized predictors to compare effect sizes
Xz = df[['rel_size','rel_loc']].copy()
Xz = (Xz - Xz.mean())/Xz.std(ddof=0)
Xz = sm.add_constant(Xz)
model_z = sm.Logit(y, Xz).fit(disp=False)
print('\nStandardized coefficients:')
print(model_z.params)
print(model_z.pvalues)

