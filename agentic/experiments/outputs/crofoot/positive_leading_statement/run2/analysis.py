import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Create predictors
df['size_diff'] = df['n_focal'] - df['n_other']
df['loc_adv'] = df['dist_other'] - df['dist_focal']  # positive means focal closer to its home range center

# Standardize predictors for comparability (optional)
X = df[['size_diff', 'loc_adv']]
X = (X - X.mean()) / X.std(ddof=0)
X = sm.add_constant(X)

y = df['win']

# Fit logistic regression
model = sm.Logit(y, X).fit(disp=False)

# Summaries
print('Logit results (standardized predictors):')
print(model.summary())

# Odds ratios for 1 SD increase
params = model.params
conf = model.conf_int()
odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print('\nOdds ratios (1 SD increase):')
for name in params.index:
    if name == 'const':
        continue
    print(f"{name}: OR={odds_ratios[name]:.2f}, 95% CI=({conf_or.loc[name,0]:.2f}, {conf_or.loc[name,1]:.2f})")

# Simple descriptive check: win rate by sign of predictors
df['size_adv'] = np.where(df['size_diff'] > 0, 'focal_larger', np.where(df['size_diff'] < 0, 'focal_smaller', 'equal'))
df['loc_adv_sign'] = np.where(df['loc_adv'] > 0, 'focal_closer', np.where(df['loc_adv'] < 0, 'other_closer', 'equal'))

print('\\nWin rates by size advantage:')
print(df.groupby('size_adv')['win'].mean())

print('\\nWin rates by location advantage:')
print(df.groupby('loc_adv_sign')['win'].mean())

# Combined descriptive table
combo = df.pivot_table(values='win', index='size_adv', columns='loc_adv_sign', aggfunc='mean')
print('\\nWin rate by size_adv x loc_adv_sign:')
print(combo)
