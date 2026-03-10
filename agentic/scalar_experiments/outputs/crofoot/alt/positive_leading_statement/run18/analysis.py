import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint


df = pd.read_csv('crofoot.csv')

# Construct predictors
# Relative group size: focal minus other
# Location advantage: positive if contest is closer to focal home range center
# (i.e., other is farther from its center than focal)

df['size_diff'] = df['n_focal'] - df['n_other']

df['loc_diff'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for easier interpretation
for col in ['size_diff', 'loc_diff']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression: win ~ size_diff + loc_diff
X = sm.add_constant(df[['size_diff_z', 'loc_diff_z']])
model = sm.Logit(df['win'], X).fit(disp=False)

# Also test each predictor separately
X_size = sm.add_constant(df[['size_diff_z']])
model_size = sm.Logit(df['win'], X_size).fit(disp=False)

X_loc = sm.add_constant(df[['loc_diff_z']])
model_loc = sm.Logit(df['win'], X_loc).fit(disp=False)

# Simple descriptive: win rate by sign of size_diff and loc_diff

def win_rate_by_sign(series, label):
    out = {}
    for sign, name in [(-1, 'negative'), (0, 'zero'), (1, 'positive')]:
        if sign == 0:
            subset = df[series == 0]
        elif sign < 0:
            subset = df[series < 0]
        else:
            subset = df[series > 0]
        if len(subset) == 0:
            continue
        win_rate = subset['win'].mean()
        ci_low, ci_high = proportion_confint(subset['win'].sum(), len(subset), method='wilson')
        out[name] = (len(subset), win_rate, ci_low, ci_high)
    return out

size_rates = win_rate_by_sign(df['size_diff'], 'size_diff')
loc_rates = win_rate_by_sign(df['loc_diff'], 'loc_diff')

print('n_rows', len(df))
print('size_diff values', df['size_diff'].describe())
print('loc_diff values', df['loc_diff'].describe())

print('\nLogit win ~ size_diff_z + loc_diff_z')
print(model.summary())

print('\nLogit win ~ size_diff_z')
print(model_size.summary())

print('\nLogit win ~ loc_diff_z')
print(model_loc.summary())

print('\nWin rates by size_diff sign')
print(size_rates)

print('\nWin rates by loc_diff sign')
print(loc_rates)

# Correlation between predictors
print('\nPredictor correlation')
print(df[['size_diff', 'loc_diff']].corr())
