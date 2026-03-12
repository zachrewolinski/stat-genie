import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived variables
df = df.copy()
df['size_diff'] = df['feature7'] - df['feature8']
df['loc_adv'] = df['feature6'] - df['feature5']  # positive = contest closer to focal center

# Binary indicators
df['focal_larger'] = df['size_diff'] > 0
df['focal_closer'] = df['loc_adv'] > 0

# Win rates by indicators
print('Win rate overall:', df['feature4'].mean())
print('\nWin rate by focal_larger:')
print(df.groupby('focal_larger')['feature4'].mean())
print('\nCounts by focal_larger:')
print(df['focal_larger'].value_counts())

print('\nWin rate by focal_closer:')
print(df.groupby('focal_closer')['feature4'].mean())
print('\nCounts by focal_closer:')
print(df['focal_closer'].value_counts())

# Logistic regression with binary predictors
X = df[['focal_larger', 'focal_closer']].astype(int)
X = sm.add_constant(X)
y = df['feature4']
model = sm.Logit(y, X)
result = model.fit(disp=False)
print('\nLogit with binary predictors:')
print(result.summary())
print('\nOdds ratios:')
print(np.exp(result.params))
print('\nP-values:')
print(result.pvalues)
