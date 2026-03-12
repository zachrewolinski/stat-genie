import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Identify variables based on metadata mapping
# Skin tone ratings
skin1 = _df['rater1']
skin2 = _df['nExp']
skin = (skin1 + skin2) / 2.0

# Red cards count and games (exposure)
red_cards = _df['yellowCards']  # per metadata: number of red cards
_games = _df['redCards']        # per metadata: number of games in dyad

# Basic cleanup
mask = (~skin.isna()) & (~red_cards.isna()) & (~_games.isna()) & (_games > 0)

df = pd.DataFrame({
    'skin': skin[mask],
    'red_cards': red_cards[mask],
    'games': _games[mask]
}).copy()

# Continuous Poisson regression with offset for exposure
X = sm.add_constant(df['skin'])
model = sm.GLM(df['red_cards'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
res = model.fit()

# Overdispersion check
pearson_chi2 = sum(res.resid_pearson ** 2)
df_resid = res.df_resid
overdisp = pearson_chi2 / df_resid if df_resid > 0 else np.nan

# Dark vs light comparison (extremes)
light = df['skin'] <= 0.25
_dark = df['skin'] >= 0.75
extreme_df = df[light | _dark].copy()

rate_light = (extreme_df.loc[light, 'red_cards'].sum() / extreme_df.loc[light, 'games'].sum()) if light.any() else np.nan
rate_dark = (extreme_df.loc[_dark, 'red_cards'].sum() / extreme_df.loc[_dark, 'games'].sum()) if _dark.any() else np.nan
rate_ratio = rate_dark / rate_light if (rate_light and rate_dark) else np.nan

# Poisson regression on extreme groups
if extreme_df['skin'].nunique() > 1:
    extreme_df['dark'] = (extreme_df['skin'] >= 0.75).astype(int)
    Xb = sm.add_constant(extreme_df['dark'])
    model_b = sm.GLM(extreme_df['red_cards'], Xb, family=sm.families.Poisson(), offset=np.log(extreme_df['games']))
    res_b = model_b.fit()
else:
    res_b = None

print('N total', len(df))
print('Red cards total', df['red_cards'].sum())
print('Games total', df['games'].sum())
print('Skin mean', df['skin'].mean(), 'min', df['skin'].min(), 'max', df['skin'].max())
print('\nPoisson regression (continuous skin):')
print(res.summary())
print('Overdispersion ratio (Pearson chi2 / df):', overdisp)

print('\nExtreme groups:')
print('N light', light.sum(), 'N dark', _dark.sum())
print('Rate light', rate_light, 'Rate dark', rate_dark, 'Rate ratio', rate_ratio)
if res_b is not None:
    print('\nPoisson regression (dark vs light):')
    print(res_b.summary())
