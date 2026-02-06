import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Keep relevant columns
cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other']
df = _df[cols].copy()

# Drop rows with missing values
before = len(df)
df = df.dropna()
after = len(df)

# Feature engineering
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']

# Location advantage: positive if focal closer to its home-range center than other is to its center
# (i.e., contest is relatively closer to focal's center)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for interpretability
for c in ['rel_size', 'loc_adv']:
    df[f'z_{c}'] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

# Logistic regression
X = sm.add_constant(df[['z_rel_size', 'z_loc_adv']])
y = df['win']

model = sm.Logit(y, X).fit(disp=False)

# Summaries
print('Rows used:', after, 'of', before)
print(model.summary())

# Odds ratios
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)

odds_table = pd.DataFrame({
    'odds_ratio': odds,
    'conf_low': conf_odds[0],
    'conf_high': conf_odds[1],
    'p_value': model.pvalues
})
print('\nOdds ratios (standardized predictors):')
print(odds_table)

# Simple win rates by sign of rel_size and loc_adv
for name, col in [('rel_size', 'rel_size'), ('loc_adv', 'loc_adv')]:
    df['sign'] = np.where(df[col] > 0, 'positive', np.where(df[col] < 0, 'negative', 'zero'))
    rates = df.groupby('sign')['win'].mean()
    counts = df['sign'].value_counts()
    print(f"\nWin rate by sign of {name}:")
    print(pd.DataFrame({'win_rate': rates, 'n': counts}).sort_index())

# Correlation
print('\nCorrelations with win:')
print(df[['win', 'rel_size', 'loc_adv']].corr()['win'])
