import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Define relative size and location advantage for the focal group
# Relative size: focal group size minus other group size
# Relative location: positive when contest is closer to focal's home range center
# (i.e., other group is farther from its own center)
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for comparable effect sizes
for col in ['rel_size', 'rel_location']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression: win ~ rel_size + rel_location
X = df[['rel_size_z', 'rel_location_z']]
X = sm.add_constant(X)
y = df['win']
model = sm.Logit(y, X).fit(disp=False)

# Odds ratios and 95% CI
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
odds_ratios = np.exp(params)
conf_odds = np.exp(conf)

print('N:', len(df))
print('Win rate:', df['win'].mean())
print('\nLogistic regression (standardized predictors):')
print(model.summary())

print('\nOdds ratios (standardized predictors):')
for term in ['rel_size_z', 'rel_location_z']:
    print(f"{term}: OR={odds_ratios[term]:.3f} (95% CI {conf_odds.loc[term, '2.5%']:.3f} to {conf_odds.loc[term, '97.5%']:.3f}), p={model.pvalues[term]:.4f}")

# Sensitivity: use raw differences (non-standardized)
X_raw = sm.add_constant(df[['rel_size', 'rel_location']])
model_raw = sm.Logit(y, X_raw).fit(disp=False)

print('\nLogistic regression (raw predictors):')
print(model_raw.summary())

# Simple descriptive check: win rate by sign of rel_size and rel_location
for col in ['rel_size', 'rel_location']:
    grp = df.groupby(df[col] > 0)['win'].mean()
    print(f"\nWin rate when {col} > 0 vs <= 0:")
    print(grp)
