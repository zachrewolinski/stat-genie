import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Column mapping (names are shuffled in this dataset)
# yellowCards = red cards count in dyad
# redCards = number of games in dyad
# rater1 and nExp = skin tone ratings (0 = very light, 1 = very dark)
red_cards = df['yellowCards']
games = df['redCards']
skin = df[['rater1', 'nExp']].mean(axis=1)

# Drop rows without skin ratings
mask = skin.notna()
df = df[mask].copy()
red_cards = df['yellowCards']
games = df['redCards']
skin = df[['rater1', 'nExp']].mean(axis=1)

# Define dark vs light using midpoint threshold
# (values > 0.5 represent darker-than-medium skin tone)
dark = (skin > 0.5).astype(int)

# Compute red-card rates per game
rate_dark = red_cards[dark == 1].sum() / games[dark == 1].sum()
rate_light = red_cards[dark == 0].sum() / games[dark == 0].sum()
rate_ratio = rate_dark / rate_light

# Poisson regression with exposure offset for games played
X = sm.add_constant(dark)
model = sm.GLM(red_cards, X, family=sm.families.Poisson(), offset=np.log(games))
res = model.fit()

coef = res.params.iloc[1]
se = res.bse.iloc[1]
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se
rr = np.exp(coef)
rr_low = np.exp(ci_low)
rr_high = np.exp(ci_high)

print('Rows with skin ratings:', len(df))
print('Red-card rate (dark):', rate_dark)
print('Red-card rate (light):', rate_light)
print('Rate ratio (dark/light):', rate_ratio)
print('Poisson rate ratio:', rr)
print('95% CI:', (rr_low, rr_high))
print('p-value:', res.pvalues.iloc[1])
