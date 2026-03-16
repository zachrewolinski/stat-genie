import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = 'soccer.csv'

df = pd.read_csv(DATA_PATH)

# Skin tone: average of rater1 and rater2 when available
skin = df[['rater1','rater2']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Filter to rows with skin tone and games > 0
fdf = df[(~df['skin_tone'].isna()) & (df['games'] > 0)].copy()

# Aggregate to player level to reduce dyad dependence
player = (
    fdf.groupby('playerShort')
    .agg(
        total_red=('redCards','sum'),
        total_games=('games','sum'),
        skin_tone=('skin_tone','mean')
    )
    .reset_index()
)
player = player[player['total_games'] > 0].copy()
player['red_rate'] = player['total_red'] / player['total_games']

# Define light vs dark using midpoint (<=0.5 vs >0.5)
player['tone_group'] = np.where(player['skin_tone'] > 0.5, 'dark', 'light')

# Summary stats
summary = player.groupby('tone_group').agg(
    n_players=('playerShort','nunique'),
    total_red=('total_red','sum'),
    total_games=('total_games','sum'),
    mean_rate=('red_rate','mean')
).reset_index()

# t-test on red_rate between groups (Welch)
light_rates = player.loc[player['tone_group']=='light','red_rate']
dark_rates = player.loc[player['tone_group']=='dark','red_rate']

t_stat, t_p = stats.ttest_ind(dark_rates, light_rates, equal_var=False, nan_policy='omit')

# Poisson regression with offset (player-level)
X = sm.add_constant(player['skin_tone'])
y = player['total_red']
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(player['total_games']))
res = model.fit(cov_type='HC0')

coef = res.params['skin_tone']
se = res.bse['skin_tone']
pval = res.pvalues['skin_tone']
rate_ratio = np.exp(coef)

# Also dyad-level Poisson with offset for robustness
X2 = sm.add_constant(fdf['skin_tone'])
y2 = fdf['redCards']
model2 = sm.GLM(y2, X2, family=sm.families.Poisson(), offset=np.log(fdf['games']))
res2 = model2.fit(cov_type='HC0')
coef2 = res2.params['skin_tone']
pval2 = res2.pvalues['skin_tone']
rate_ratio2 = np.exp(coef2)

print('PLAYER_LEVEL_SUMMARY')
print(summary.to_string(index=False))
print('\nTTEST_RED_RATE_DARK_VS_LIGHT')
print({'t_stat': float(t_stat), 'p_value': float(t_p), 'dark_mean_rate': float(dark_rates.mean()), 'light_mean_rate': float(light_rates.mean())})
print('\nPOISSON_PLAYER_LEVEL')
print({'coef_skin': float(coef), 'se': float(se), 'p_value': float(pval), 'rate_ratio_per_1unit': float(rate_ratio)})
print('\nPOISSON_DYAD_LEVEL')
print({'coef_skin': float(coef2), 'p_value': float(pval2), 'rate_ratio_per_1unit': float(rate_ratio2)})
