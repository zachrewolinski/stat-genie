import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# Skin tone average (0-1 scale)
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Keep rows with skin tone and games
base = df.dropna(subset=['skin_tone', 'games', 'redCards']).copy()

# Aggregate at player level
player = (
    base.groupby('playerShort', as_index=False)
      .agg(
          skin_tone=('skin_tone','mean'),
          games_total=('games','sum'),
          redCards_total=('redCards','sum'),
          height=('height','mean'),
          weight=('weight','mean')
      )
)
player = player[player['games_total'] > 0].copy()
player['red_rate'] = player['redCards_total'] / player['games_total']

# Poisson GLM with offset (player level)
X = sm.add_constant(player['skin_tone'])
model = sm.GLM(player['redCards_total'], X, family=sm.families.Poisson(), offset=np.log(player['games_total']))
res = model.fit(cov_type='HC0')

# Dispersion
player_dispersion = res.deviance / res.df_resid

# Quantile-based dark vs light
q25 = player['skin_tone'].quantile(0.25)
q75 = player['skin_tone'].quantile(0.75)
light = player[player['skin_tone'] <= q25]
dark = player[player['skin_tone'] >= q75]
light_rate = light['redCards_total'].sum() / light['games_total'].sum()
dark_rate = dark['redCards_total'].sum() / dark['games_total'].sum()
rate_ratio = dark_rate / light_rate if light_rate > 0 else np.nan

# Dyad-level Poisson with controls
# Controls: games exposure, leagueCountry, position, height, weight, goals, yellowCards, yellowReds, victories, ties, defeats
controls = [
    'games','height','weight','goals','yellowCards','yellowReds','victories','ties','defeats'
]

control_df = base.dropna(subset=controls + ['leagueCountry','position']).copy()

# Build design matrix
cat = pd.get_dummies(control_df[['leagueCountry','position']], drop_first=True)
Xc = pd.concat([control_df[['skin_tone'] + controls], cat], axis=1)
Xc = sm.add_constant(Xc, has_constant='add')

offset = np.log(control_df['games'])

model_c = sm.GLM(control_df['redCards'], Xc, family=sm.families.Poisson(), offset=offset)
res_c = model_c.fit(cov_type='HC0')

control_dispersion = res_c.deviance / res_c.df_resid

# Logistic on dyads: any red card vs skin tone + games
control_df['any_red'] = (control_df['redCards'] > 0).astype(int)
X2 = sm.add_constant(control_df[['skin_tone','games']])
logit = sm.Logit(control_df['any_red'], X2)
try:
    logit_res = logit.fit(disp=False)
except Exception:
    logit_res = logit.fit(disp=False, method='lbfgs', maxiter=200)


print('n_dyads_total', len(base))
print('n_players', len(player))
print('player_poisson_coef', res.params['skin_tone'])
print('player_poisson_p', res.pvalues['skin_tone'])
print('player_poisson_IRR', np.exp(res.params['skin_tone']))
print('player_dispersion', player_dispersion)
print('light_rate', light_rate)
print('dark_rate', dark_rate)
print('rate_ratio_dark_light', rate_ratio)

print('control_dyads', len(control_df))
print('dyad_poisson_coef', res_c.params['skin_tone'])
print('dyad_poisson_p', res_c.pvalues['skin_tone'])
print('dyad_poisson_IRR', np.exp(res_c.params['skin_tone']))
print('dyad_dispersion', control_dispersion)

print('logit_coef', logit_res.params['skin_tone'])
print('logit_p', logit_res.pvalues['skin_tone'])
print('logit_OR', np.exp(logit_res.params['skin_tone']))

