import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DF = pd.read_csv('soccer.csv')

# Identify variables by value patterns
# Skin ratings: rater1 (0-1 in steps) and nExp (0-1 in steps)
skin1 = DF['rater1']
skin2 = DF['nExp']

# Red cards: column with values 0-3 (meanExp)
red_cards = DF['meanExp']

# Exposure (games per dyad): column with min 1, max 47 (redCards)
games = DF['redCards']

# Build analysis frame
analysis = pd.DataFrame({
    'skin1': skin1,
    'skin2': skin2,
    'red_cards': red_cards,
    'games': games,
})

# Drop missing skin tone data
analysis = analysis.dropna(subset=['skin1', 'skin2'])
analysis['skin_mean'] = (analysis['skin1'] + analysis['skin2']) / 2.0

# Define light/dark groups (exclude mid tone = 0.5)
analysis['skin_group'] = np.where(analysis['skin_mean'] > 0.5, 'dark',
                          np.where(analysis['skin_mean'] < 0.5, 'light', 'mid'))

# Group stats
analysis['red_per_game'] = analysis['red_cards'] / analysis['games']

summary = analysis.groupby('skin_group').agg(
    n=('red_cards', 'size'),
    red_cards_mean=('red_cards', 'mean'),
    games_mean=('games', 'mean'),
    red_per_game_mean=('red_per_game', 'mean'),
    red_cards_sum=('red_cards', 'sum'),
    games_sum=('games', 'sum'),
)

summary['rate_overall'] = summary['red_cards_sum'] / summary['games_sum']
print('Group summary:\n', summary)

# Poisson regression with offset: red_cards ~ skin_mean
analysis = analysis[analysis['games'] > 0].copy()

X = sm.add_constant(analysis['skin_mean'])
model = sm.GLM(analysis['red_cards'], X, family=sm.families.Poisson(),
               offset=np.log(analysis['games']))
res = model.fit(cov_type='HC3')
print('\nPoisson GLM (rate) with skin_mean as continuous predictor:')
print(res.summary())

# Poisson with binary dark vs light (exclude mid)
bin_df = analysis[analysis['skin_group'].isin(['light', 'dark'])].copy()
bin_df['dark'] = (bin_df['skin_group'] == 'dark').astype(int)
Xb = sm.add_constant(bin_df['dark'])
model_b = sm.GLM(bin_df['red_cards'], Xb, family=sm.families.Poisson(),
                 offset=np.log(bin_df['games']))
res_b = model_b.fit(cov_type='HC3')
print('\nPoisson GLM (rate) with dark indicator:')
print(res_b.summary())

# Effect size: rate ratio dark vs light
rate_light = summary.loc['light', 'rate_overall'] if 'light' in summary.index else np.nan
rate_dark = summary.loc['dark', 'rate_overall'] if 'dark' in summary.index else np.nan
print('\nRate overall - light:', rate_light, 'dark:', rate_dark, 'ratio:', rate_dark / rate_light if rate_light else np.nan)

# Sensitivity: if red_cards is actually yellowCards (0-2) instead of meanExp
alt_red = DF['yellowCards']
alt = pd.DataFrame({
    'skin1': skin1,
    'skin2': skin2,
    'red_cards': alt_red,
    'games': games,
}).dropna(subset=['skin1','skin2'])
alt['skin_mean'] = (alt['skin1'] + alt['skin2']) / 2.0
alt['skin_group'] = np.where(alt['skin_mean'] > 0.5, 'dark',
                      np.where(alt['skin_mean'] < 0.5, 'light', 'mid'))
alt['red_per_game'] = alt['red_cards'] / alt['games']
alt_sum = alt.groupby('skin_group').agg(
    red_cards_sum=('red_cards','sum'),
    games_sum=('games','sum'),
    n=('red_cards','size'),
)
alt_sum['rate_overall'] = alt_sum['red_cards_sum'] / alt_sum['games_sum']
print('\nAlt red card (yellowCards) overall rates:\n', alt_sum)

# Poisson for alt red
alt = alt[alt['games'] > 0]
X_alt = sm.add_constant(alt['skin_mean'])
res_alt = sm.GLM(alt['red_cards'], X_alt, family=sm.families.Poisson(),
                 offset=np.log(alt['games'])).fit(cov_type='HC3')
print('\nAlt Poisson GLM (rate) with skin_mean as continuous predictor:')
print(res_alt.summary())
