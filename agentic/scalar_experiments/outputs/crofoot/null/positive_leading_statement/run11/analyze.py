import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create relative group size and location variables
# Relative group size: focal minus other
# Contest location: distance to focal center minus distance to other center
# (positive means contest closer to other center; negative means closer to focal center)
df['size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Logistic regression
X = df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
y = df['win']

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

print(result.summary())

# Also compute simple effect sizes: mean win by size advantage and by location advantage
# Size advantage: focal larger (size_diff > 0), equal, smaller
size_bins = pd.cut(df['size_diff'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=['smaller', 'equal', 'larger'])
print('\nWin rate by size advantage:')
print(df.groupby(size_bins)['win'].mean())

# Location advantage: contest closer to focal (dist_diff < 0) vs closer to other (dist_diff > 0)
loc_bins = pd.cut(df['dist_diff'], bins=[-np.inf, -1e-6, 1e-6, np.inf], labels=['closer_focal', 'equal', 'closer_other'])
print('\nWin rate by location advantage:')
print(df.groupby(loc_bins)['win'].mean())

# Correlation check
print('\nCorrelation size_diff and dist_diff:', df['size_diff'].corr(df['dist_diff']))
