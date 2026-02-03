import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Compute average skin tone rating
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Red-card rate per game in each dyad
df['red_per_game'] = df['redCards'] / df['games']

# Compare light vs dark using top/bottom quartiles
q1 = df['skin_tone'].quantile(0.25)
q3 = df['skin_tone'].quantile(0.75)
sub = df[(df['skin_tone'] <= q1) | (df['skin_tone'] >= q3)].copy()
sub['dark'] = sub['skin_tone'] >= q3

rate_by_group = sub.groupby('dark').apply(
    lambda g: g['redCards'].sum() / g['games'].sum()
)

# Poisson regression of red card counts with exposure (games)
X = sm.add_constant(df['skin_tone'])
poisson_model = sm.GLM(
    df['redCards'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(df['games'])
).fit()

print('Skin tone quartiles:', q1, q3)
print('Red-card rate per game (light=bottom quartile, dark=top quartile):')
print(rate_by_group)
print('\nPoisson regression (offset log(games))')
print(poisson_model.summary().tables[1])
