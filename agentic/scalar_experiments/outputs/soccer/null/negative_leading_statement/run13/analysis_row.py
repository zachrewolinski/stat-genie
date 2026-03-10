import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Row-level skin mean
skin_mean = df[['rater1','rater2']].mean(axis=1)
df = df.assign(skin_mean=skin_mean)

# Keep rows with skin rating and games > 0
base = df[(df['skin_mean'].notna()) & (df['games'] > 0)].copy()

# Binary dark/light definition for contrast (extremes)
base['skin_group'] = np.where(base['skin_mean'] <= 0.25, 'light', np.where(base['skin_mean'] >= 0.75, 'dark', 'mid'))

# Summary rates by group (dyad-level totals)
summary = (
    base.groupby('skin_group')
    .agg(dyads=('playerShort','size'),
         players=('playerShort','nunique'),
         total_games=('games','sum'),
         total_red=('redCards','sum'))
    .assign(red_per_game=lambda x: x['total_red'] / x['total_games'])
)

# Poisson GLM on dyads (binary dark vs light, excluding mid)
bin_df = base[base['skin_group'].isin(['dark','light'])].copy()

# Add log offset
bin_df['log_games'] = np.log(bin_df['games'])

# Simple model: redCards ~ dark + offset(log(games))
# Use robust SE due to overdispersion
bin_df['dark'] = (bin_df['skin_group'] == 'dark').astype(int)
model_simple = sm.GLM(bin_df['redCards'], sm.add_constant(bin_df['dark']),
                      family=sm.families.Poisson(), offset=bin_df['log_games']).fit(cov_type='HC1')

# Adjusted model with covariates (position, leagueCountry, height, weight)
adj_cols = ['redCards','dark','log_games','position','leagueCountry','height','weight']
adj_df = bin_df[adj_cols].dropna().copy()
formula = 'redCards ~ dark + C(position) + C(leagueCountry) + height + weight'
model_adj = smf.glm(formula=formula, data=adj_df,
                    family=sm.families.Poisson(), offset=adj_df['log_games']).fit(cov_type='HC1')

# Continuous skin tone model (dyad-level)
cont_df = base[['redCards','skin_mean','games']].copy().dropna()
cont_df['log_games'] = np.log(cont_df['games'])
model_cont = sm.GLM(cont_df['redCards'], sm.add_constant(cont_df['skin_mean']),
                    family=sm.families.Poisson(), offset=cont_df['log_games']).fit(cov_type='HC1')

# Player-level aggregation using row-level skin mean (average per player)
player = (
    base.groupby('playerShort')
    .agg(skin_mean=('skin_mean','mean'),
         total_games=('games','sum'),
         total_red=('redCards','sum'))
    .reset_index()
)
player = player[player['total_games'] > 0]
player['log_games'] = np.log(player['total_games'])

# Define player-level dark/light
player['skin_group'] = np.where(player['skin_mean'] <= 0.25, 'light', np.where(player['skin_mean'] >= 0.75, 'dark', 'mid'))
player_bin = player[player['skin_group'].isin(['dark','light'])].copy()
player_bin['dark'] = (player_bin['skin_group'] == 'dark').astype(int)
player_model = sm.GLM(player_bin['total_red'], sm.add_constant(player_bin['dark']),
                      family=sm.families.Poisson(), offset=player_bin['log_games']).fit(cov_type='HC1')

# Correlation on continuous skin tone vs red card rate (player-level)
player['red_per_game'] = player['total_red'] / player['total_games']

corr, corr_p = spearmanr(player['skin_mean'], player['red_per_game'], nan_policy='omit')

# Print key results
print('SUMMARY_BY_GROUP')
print(summary)
print('\nDYAD_SIMPLE_MODEL')
print(model_simple.summary().tables[1])
print('\nDYAD_ADJ_MODEL')
print(model_adj.summary().tables[1])
print('\nDYAD_CONT_MODEL')
print(model_cont.summary().tables[1])
print('\nPLAYER_MODEL')
print(player_model.summary().tables[1])
print('\nSPEARMAN_PLAYER_LEVEL')
print({'spearman_r': corr, 'p_value': corr_p})
