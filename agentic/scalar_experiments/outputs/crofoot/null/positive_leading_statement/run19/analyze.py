import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('crofoot.csv')

# Prepare predictors

df['size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_other'] - df['dist_focal']  # positive means other farther from its home range center than focal

# Logistic regression

X = df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())

# Alternative: relative location as which group is closer

df['focal_closer'] = (df['dist_focal'] < df['dist_other']).astype(int)

# Logistic with focal_closer

X2 = df[['size_diff', 'focal_closer']]
X2 = sm.add_constant(X2)

model2 = sm.Logit(y, X2)
result2 = model2.fit(disp=False)

print('\nModel with focal_closer:')
print(result2.summary())

# Quick descriptive stats

print('\nWin rate by size_diff:')
print(df.groupby('size_diff')['win'].mean())

print('\nWin rate by focal_closer:')
print(df.groupby('focal_closer')['win'].mean())

# Correlation tests (point-biserial)

r_size, p_size = stats.pointbiserialr(df['win'], df['size_diff'])
r_dist, p_dist = stats.pointbiserialr(df['win'], df['dist_diff'])

print('\nPoint-biserial correlations:')
print('size_diff r=%.3f p=%.4f' % (r_size, p_size))
print('dist_diff r=%.3f p=%.4f' % (r_dist, p_dist))

