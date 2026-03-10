import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data

df = pd.read_csv('crofoot.csv')

# Rename for clarity

df = df.rename(columns={
    'feature4': 'focal_win',
    'feature5': 'focal_dist',
    'feature6': 'other_dist',
    'feature7': 'focal_size',
    'feature8': 'other_size'
})

# Relative group size and relative location

df['size_diff'] = df['focal_size'] - df['other_size']
df['size_ratio'] = df['focal_size'] / df['other_size']
df['dist_diff'] = df['focal_dist'] - df['other_dist']

# Logistic regression with size_diff and dist_diff

X = df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
y = df['focal_win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also test size_ratio with dist_diff to check robustness

X2 = df[['size_ratio', 'dist_diff']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2)
result2 = model2.fit(disp=False)

# Output summary metrics

print('Model 1 (size_diff, dist_diff)')
print(result.summary2().tables[1])
print('\nModel 2 (size_ratio, dist_diff)')
print(result2.summary2().tables[1])

# Simple descriptive: win rate by size_diff sign and dist_diff sign

df['size_adv'] = np.where(df['size_diff'] > 0, 'larger', np.where(df['size_diff'] < 0, 'smaller', 'equal'))
df['location_adv'] = np.where(df['dist_diff'] < 0, 'focal_closer', np.where(df['dist_diff'] > 0, 'other_closer', 'equal'))

print('\nWin rate by size_adv')
print(df.groupby('size_adv')['focal_win'].mean())
print('\nWin rate by location_adv')
print(df.groupby('location_adv')['focal_win'].mean())

print('\nCounts')
print(df[['size_adv', 'location_adv']].value_counts())
