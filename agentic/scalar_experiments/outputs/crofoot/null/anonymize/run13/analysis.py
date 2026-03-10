import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Define variables
df['win'] = df['feature4']
# Relative group size (focal - other)
df['rel_size'] = df['feature7'] - df['feature8']
# Location advantage: positive if contest is closer to focal group's home range center
# (other group's distance from its center minus focal group's distance from its center)
df['loc_adv'] = df['feature6'] - df['feature5']

# Basic sanity
print('Rows:', len(df))
print('Win rate:', df['win'].mean())
print('Rel size summary:', df['rel_size'].describe())
print('Loc adv summary:', df['loc_adv'].describe())

# Fit logistic regression with both predictors
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X).fit(disp=False)
print('\nLogit with rel_size and loc_adv')
print(model.summary())

# Univariate models for each predictor
for var in ['rel_size', 'loc_adv']:
    X1 = sm.add_constant(df[[var]])
    m1 = sm.Logit(y, X1).fit(disp=False)
    print(f"\nUnivariate logit: {var}")
    print(m1.summary())

# Also fit scaled predictors to compare effect sizes
scaled = df[['rel_size', 'loc_adv']].copy()
scaled = (scaled - scaled.mean()) / scaled.std(ddof=0)
Xz = sm.add_constant(scaled)
model_z = sm.Logit(y, Xz).fit(disp=False)
print('\nLogit with standardized predictors')
print(model_z.summary())

# Save key stats to a json-like text for quick view
stats = {
    'params': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'params_std': model_z.params.to_dict(),
    'pvalues_std': model_z.pvalues.to_dict(),
}
print('\nKey stats:', stats)
