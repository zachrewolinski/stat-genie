import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Construct predictors
# Relative group size (difference and ratio)
df['size_diff'] = df['n_focal'] - df['n_other']
df['size_log_ratio'] = np.log(df['n_focal'] / df['n_other'])

# Contest location: relative distance to home range centers
# Positive means contest closer to focal group's center than to other group's center
df['loc_diff'] = df['dist_other'] - df['dist_focal']

# Logistic regression with size_diff and loc_diff
X = df[['size_diff', 'loc_diff']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X).fit(disp=False)

# Alternative with size_log_ratio
X2 = df[['size_log_ratio', 'loc_diff']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(df['win'], X2).fit(disp=False)

# Summaries
print('Model 1: win ~ size_diff + loc_diff')
print(model.summary())
print('\nModel 2: win ~ size_log_ratio + loc_diff')
print(model2.summary())

# Marginal effect sense (odds ratios)
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

print('\nOdds ratios (Model 1):')
for name in params.index:
    print(f"{name}: OR={odds[name]:.3f}, 95% CI=({conf_odds.loc[name,0]:.3f}, {conf_odds.loc[name,1]:.3f}), p={model.pvalues[name]:.4f}")

# Descriptive stats
print('\nWin rate overall:', df['win'].mean())
print('Win rate by size_diff:')
print(df.groupby('size_diff')['win'].mean())
print('\nCorrelation (size_diff, win):', df['size_diff'].corr(df['win']))
print('Correlation (loc_diff, win):', df['loc_diff'].corr(df['win']))
