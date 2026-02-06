import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Create relative size and relative location measures
# Relative size: focal group size minus other group size
# Relative location: distance of other minus distance of focal (positive => focal closer to its center)
df['size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_other'] - df['dist_focal']

# Basic summaries
summary = {
    'n': len(df),
    'win_rate': df['win'].mean(),
    'size_diff_mean': df['size_diff'].mean(),
    'dist_diff_mean': df['dist_diff'].mean(),
}

# Logistic regression: win ~ size_diff + dist_diff
X = df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
y = df['win']
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

# Also check model with size ratio and relative location (robustness)
df['size_ratio'] = df['n_focal'] / df['n_other']
X2 = sm.add_constant(df[['size_ratio', 'dist_diff']])
logit_model2 = sm.Logit(y, X2)
result2 = logit_model2.fit(disp=False)

# Simple win rates by advantage
size_adv_win = df.loc[df['size_diff'] > 0, 'win'].mean()
size_disadv_win = df.loc[df['size_diff'] < 0, 'win'].mean()
loc_adv_win = df.loc[df['dist_diff'] > 0, 'win'].mean()
loc_disadv_win = df.loc[df['dist_diff'] < 0, 'win'].mean()

# Output results
print('SUMMARY', summary)
print('\nLOGIT: win ~ size_diff + dist_diff')
print(result.summary())
print('\nLOGIT: win ~ size_ratio + dist_diff')
print(result2.summary())
print('\nWin rates:')
print('  size advantage (size_diff>0):', size_adv_win)
print('  size disadvantage (size_diff<0):', size_disadv_win)
print('  location advantage (dist_diff>0):', loc_adv_win)
print('  location disadvantage (dist_diff<0):', loc_disadv_win)
