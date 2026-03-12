import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'soccer.csv'

df = pd.read_csv(path)

# compute skin tone average
for col in ['rater1','rater2']:
    if col not in df.columns:
        raise SystemExit(f'missing {col}')

df['skin_tone'] = df[['rater1','rater2']].mean(axis=1)

# dyad-level subset
sub = df[['playerShort','skin_tone','redCards','games']].dropna()

print('DYAD-LEVEL')
print('rows', len(sub))
print('redCards sum', sub['redCards'].sum())
print('games sum', sub['games'].sum())

# define light/medium/dark (quartile-ish scale)
sub['group'] = pd.cut(sub['skin_tone'], bins=[-1,0.25,0.75,2], labels=['light','medium','dark'])
print('3-level group counts')
print(sub['group'].value_counts())

agg = sub.groupby('group').agg(redCards=('redCards','sum'), games=('games','sum'), dyads=('redCards','size'))
agg['rate_per_game'] = agg['redCards'] / agg['games']
print('3-level group rates')
print(agg)

# Binary group using >=0.5 as darker
sub['dark_05'] = (sub['skin_tone'] >= 0.5).astype(int)
agg2 = sub.groupby('dark_05').agg(redCards=('redCards','sum'), games=('games','sum'), dyads=('redCards','size'))
agg2['rate_per_game'] = agg2['redCards'] / agg2['games']
print('binary (>=0.5) group rates')
print(agg2)

# Poisson regression with offset log(games) using continuous skin tone
sub_poisson = sub.copy()
sub_poisson['log_games'] = np.log(sub_poisson['games'])
sub_poisson['skin_tone_c'] = sub_poisson['skin_tone'] - sub_poisson['skin_tone'].mean()

model = smf.glm('redCards ~ skin_tone_c', data=sub_poisson, family=sm.families.Poisson(), offset=sub_poisson['log_games']).fit()
print('Poisson (continuous skin tone)')
print(model.summary())

pearson_chi2 = model.pearson_chi2
ratio = pearson_chi2 / model.df_resid
print('overdispersion_ratio', ratio)

# Negative binomial (continuous)
try:
    nb_model = smf.glm('redCards ~ skin_tone_c', data=sub_poisson, family=sm.families.NegativeBinomial(alpha=1.0), offset=sub_poisson['log_games']).fit()
    print('NegBin (continuous skin tone)')
    print(nb_model.summary())
except Exception as e:
    print('NB model error', e)

# Poisson with binary dark indicator
model_bin = smf.glm('redCards ~ dark_05', data=sub_poisson, family=sm.families.Poisson(), offset=sub_poisson['log_games']).fit()
print('Poisson (binary dark>=0.5)')
print(model_bin.summary())

# PLAYER-LEVEL
print('\nPLAYER-LEVEL')
player = (
    sub.groupby('playerShort')
    .agg(skin_tone=('skin_tone','mean'), redCards=('redCards','sum'), games=('games','sum'))
    .reset_index()
)
player['rate_per_game'] = player['redCards'] / player['games']

# Binary group using >=0.5
player['dark_05'] = (player['skin_tone'] >= 0.5).astype(int)
agg_p = player.groupby('dark_05').agg(redCards=('redCards','sum'), games=('games','sum'), players=('redCards','size'))
agg_p['rate_per_game'] = agg_p['redCards'] / agg_p['games']
print('player-level binary (>=0.5) group rates')
print(agg_p)

# Poisson regression at player level with offset
player['log_games'] = np.log(player['games'])
player['skin_tone_c'] = player['skin_tone'] - player['skin_tone'].mean()
model_p = smf.glm('redCards ~ skin_tone_c', data=player, family=sm.families.Poisson(), offset=player['log_games']).fit()
print('player-level Poisson (continuous)')
print(model_p.summary())

