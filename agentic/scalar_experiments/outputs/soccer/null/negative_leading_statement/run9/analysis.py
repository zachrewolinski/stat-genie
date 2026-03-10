import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('soccer.csv')

# Helper to get modal non-null value

def mode_nonnull(series):
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.mode().iat[0]

# Compute per-player skin ratings using modal rater values
player_ratings = (
    _df.groupby('playerShort', as_index=False)
        .agg(rater1=('rater1', mode_nonnull),
             rater2=('rater2', mode_nonnull))
)
player_ratings['skin_mean'] = player_ratings[['rater1', 'rater2']].mean(axis=1, skipna=True)

# Merge back to dyads
_df = _df.merge(player_ratings[['playerShort', 'skin_mean']], on='playerShort', how='left')

# Keep rows with skin ratings and games>0
_df = _df[_df['skin_mean'].notna()].copy()
_df = _df[_df['games'] > 0].copy()

# Aggregate to player level for rates
player = (
    _df.groupby('playerShort', as_index=False)
      .agg(
          skin_mean=('skin_mean', 'mean'),
          games=('games', 'sum'),
          redCards=('redCards', 'sum')
      )
)

# Quartile groups for more balanced comparison
q25 = player['skin_mean'].quantile(0.25)
q75 = player['skin_mean'].quantile(0.75)

player['quartile_group'] = np.where(player['skin_mean'] <= q25, 'light_q1',
                            np.where(player['skin_mean'] >= q75, 'dark_q4', 'mid'))

rate_quartiles = (
    player[player['quartile_group'].isin(['light_q1', 'dark_q4'])]
      .groupby('quartile_group')
      .apply(lambda g: pd.Series({
          'players': g.shape[0],
          'games': g['games'].sum(),
          'red_cards': g['redCards'].sum(),
          'red_per_game': g['redCards'].sum() / g['games'].sum()
      }))
)

# Poisson regression at player level with offset
player['log_games'] = np.log(player['games'])

poisson_player = smf.glm(
    'redCards ~ skin_mean',
    data=player,
    family=sm.families.Poisson(),
    offset=player['log_games']
).fit()

# Negative binomial for robustness
nb_player = smf.glm(
    'redCards ~ skin_mean',
    data=player,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=player['log_games']
).fit()

# Dyad-level Poisson with clustered SE by player
_df['log_games'] = np.log(_df['games'])
poisson_dyad = smf.glm(
    'redCards ~ skin_mean',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_games']
).fit(cov_type='cluster', cov_kwds={'groups': _df['playerShort']})

# Poisson on quartile groups
quart = player[player['quartile_group'].isin(['light_q1', 'dark_q4'])].copy()
quart['log_games'] = np.log(quart['games'])

poisson_quart = smf.glm(
    'redCards ~ C(quartile_group)',
    data=quart,
    family=sm.families.Poisson(),
    offset=quart['log_games']
).fit()

# Extract key stats

def coef_summary(model, term='skin_mean'):
    coef = model.params[term]
    se = model.bse[term]
    p = model.pvalues[term]
    irr = np.exp(coef)
    return coef, se, p, irr

player_coef = coef_summary(poisson_player)
nb_coef = coef_summary(nb_player)
dyad_coef = coef_summary(poisson_dyad)
quart_coef = coef_summary(poisson_quart, 'C(quartile_group)[T.light_q1]')

print('Players with skin ratings:', player.shape[0])
print('Skin_mean quartiles:', {'q25': q25, 'q75': q75})
print('\nQuartile group rates (player level):')
print(rate_quartiles)

print('\nPoisson player-level (offset log games) skin_mean coef, se, p, IRR:')
print(player_coef)

print('\nNegative binomial player-level (offset log games) skin_mean coef, se, p, IRR:')
print(nb_coef)

print('\nPoisson dyad-level clustered SE skin_mean coef, se, p, IRR:')
print(dyad_coef)

print('\nPoisson quartile group coef (light_q1 vs dark_q4) coef, se, p, IRR:')
print(quart_coef)
