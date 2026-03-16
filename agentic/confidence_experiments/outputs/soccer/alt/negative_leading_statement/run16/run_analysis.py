import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('soccer.csv')

# Skin tone average
df['skin'] = df[['rater1','rater2']].mean(axis=1)

# Keep rows with skin and valid games
df = df[df['skin'].notna() & df['games'].gt(0)].copy()

# Rate per game
df['red_per_game'] = df['redCards'] / df['games']

# Define light/dark using scale endpoints
df['skin_group'] = np.where(df['skin'] <= 0.25, 'light', np.where(df['skin'] >= 0.75, 'dark', 'mid'))

# Summary by group
summary = df.groupby('skin_group').agg(
    n=('redCards','size'),
    games=('games','sum'),
    redCards=('redCards','sum'),
    mean_red_per_game=('red_per_game','mean'),
    any_red=('redCards', lambda s: (s>0).mean())
).reset_index()

print('Group summary (dyads):')
print(summary)

# Rate ratio dark vs light using Poisson regression with offset
df_dl = df[df['skin_group'].isin(['light','dark'])].copy()
df_dl['dark'] = (df_dl['skin_group'] == 'dark').astype(int)

# Poisson GLM for counts with exposure (games)
model = smf.glm('redCards ~ dark', data=df_dl,
                 family=sm.families.Poisson(),
                 offset=np.log(df_dl['games']))
res = model.fit(cov_type='HC0')

print('\nPoisson GLM (dark vs light), dyads:')
print(res.summary())

# Continuous skin effect (0-1)
model_cont = smf.glm('redCards ~ skin', data=df,
                      family=sm.families.Poisson(),
                      offset=np.log(df['games']))
res_cont = model_cont.fit(cov_type='HC0')
print('\nPoisson GLM (skin continuous), dyads:')
print(res_cont.summary())

# Player-level aggregation
player = df.groupby('playerShort').agg(
    skin=('skin','mean'),
    games=('games','sum'),
    redCards=('redCards','sum')
).reset_index()
player['red_per_game'] = player['redCards'] / player['games']
player['skin_group'] = np.where(player['skin'] <= 0.25, 'light', np.where(player['skin'] >= 0.75, 'dark', 'mid'))

summary_player = player.groupby('skin_group').agg(
    n_players=('playerShort','size'),
    games=('games','sum'),
    redCards=('redCards','sum'),
    mean_red_per_game=('red_per_game','mean'),
    any_red=('redCards', lambda s: (s>0).mean())
).reset_index()

print('\nGroup summary (player-level):')
print(summary_player)

player_dl = player[player['skin_group'].isin(['light','dark'])].copy()
player_dl['dark'] = (player_dl['skin_group'] == 'dark').astype(int)

model_p = smf.glm('redCards ~ dark', data=player_dl,
                  family=sm.families.Poisson(),
                  offset=np.log(player_dl['games']))
res_p = model_p.fit(cov_type='HC0')

print('\nPoisson GLM (dark vs light), player-level:')
print(res_p.summary())

# Continuous skin effect player-level
model_p_cont = smf.glm('redCards ~ skin', data=player,
                       family=sm.families.Poisson(),
                       offset=np.log(player['games']))
res_p_cont = model_p_cont.fit(cov_type='HC0')
print('\nPoisson GLM (skin continuous), player-level:')
print(res_p_cont.summary())

# Save key metrics
output = {
    'dyad_group_summary': summary,
    'player_group_summary': summary_player,
    'dyad_dark_coef': res.params.get('dark'),
    'dyad_dark_pvalue': res.pvalues.get('dark'),
    'dyad_dark_rr': float(np.exp(res.params.get('dark'))),
    'dyad_skin_coef': res_cont.params.get('skin'),
    'dyad_skin_pvalue': res_cont.pvalues.get('skin'),
    'player_dark_coef': res_p.params.get('dark'),
    'player_dark_pvalue': res_p.pvalues.get('dark'),
    'player_dark_rr': float(np.exp(res_p.params.get('dark'))),
    'player_skin_coef': res_p_cont.params.get('skin'),
    'player_skin_pvalue': res_p_cont.pvalues.get('skin'),
}

# print concise key metrics
print('\nKey metrics:')
for k, v in output.items():
    if hasattr(v, 'to_string'):
        continue
    print(k, v)
