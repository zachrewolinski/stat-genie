import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Infer column meanings based on value ranges
# m_focal appears to be binary outcome (0/1): focal wins
# f_other and win look like group sizes (5-13)
# m_other and n_focal look like distances (55-853)

# Construct predictors

df['size_diff'] = df['f_other'] - df['win']
# Positive means focal group larger than other group

df['size_ratio'] = df['f_other'] / df['win']

# Location advantage: positive when focal is closer to its own range center
# than the other group is to its range center

df['loc_diff'] = df['n_focal'] - df['m_other']
# Positive means other is farther from its own center (focal relatively advantaged)

df['loc_adv'] = (df['loc_diff'] > 0).astype(int)

# Outcome

y = df['m_focal']

# Logistic regression with size_diff and loc_diff
X = df[['size_diff', 'loc_diff']]
X = sm.add_constant(X)
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also test with size_ratio and loc_diff
X2 = df[['size_ratio', 'loc_diff']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(y, X2)
result2 = model2.fit(disp=False)

# Simple descriptive checks
win_rate = y.mean()
size_diff_mean = df['size_diff'].mean()
loc_diff_mean = df['loc_diff'].mean()

# Grouped win rates
win_by_size_adv = df.groupby(df['size_diff'] > 0)['m_focal'].mean()
win_by_loc_adv = df.groupby(df['loc_diff'] > 0)['m_focal'].mean()

print('Rows:', len(df))
print('Overall focal win rate:', win_rate)
print('Mean size_diff (focal - other):', size_diff_mean)
print('Mean loc_diff (other - focal distance):', loc_diff_mean)
print('\nWin rate by size advantage (size_diff > 0):')
print(win_by_size_adv)
print('\nWin rate by location advantage (loc_diff > 0):')
print(win_by_loc_adv)
print('\nLogit: win ~ size_diff + loc_diff')
print(result.summary())
print('\nLogit: win ~ size_ratio + loc_diff')
print(result2.summary())
