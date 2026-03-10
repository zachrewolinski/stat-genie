import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data

df = pd.read_csv('crofoot.csv')

# Map variables based on metadata checks
# Outcome: m_focal (1 focal wins, 0 other wins)
# Focal group size: f_other
# Other group size: win
# Distances from home range centers: m_other (focal), n_focal (other)

# Derived predictors

df['rel_size'] = df['f_other'] - df['win']
# Location advantage: positive means focal group is closer to its home-range center than the other group is to its own
# i.e., contest is more central to the focal group

df['loc_adv'] = df['n_focal'] - df['m_other']

# Fit logistic regression
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
model = sm.Logit(df['m_focal'], X).fit(disp=False)
print(model.summary())

# Also fit model with standardized predictors for effect comparability
X_std = df[['rel_size', 'loc_adv']].copy()
X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
X_std = sm.add_constant(X_std)
model_std = sm.Logit(df['m_focal'], X_std).fit(disp=False)
print('\nStandardized predictors model:')
print(model_std.summary())

# Simple descriptive stats
print('\nOutcome by loc_adv sign:')
print(pd.crosstab(df['m_focal'], df['loc_adv'] > 0, normalize='columns'))

print('\nOutcome by rel_size sign:')
print(pd.crosstab(df['m_focal'], df['rel_size'] > 0, normalize='columns'))

# correlation
print('\nCorrelation:')
print(df[['m_focal','rel_size','loc_adv']].corr())

