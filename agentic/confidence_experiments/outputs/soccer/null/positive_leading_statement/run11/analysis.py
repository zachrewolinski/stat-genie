import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('soccer.csv')

# Compute skin tone as mean of raters
# Some rows may have missing rater values

df['skinTone'] = df[['rater1','rater2']].mean(axis=1)

# Keep rows with skin tone and games > 0

df = df[df['skinTone'].notna() & (df['games'] > 0)]

# Basic summary
print('Rows after skin tone & games>0:', len(df))
print('Skin tone value counts:')
print(df['skinTone'].value_counts().sort_index())

# Define dark vs light: light < 0.5, dark >= 0.5

df['dark'] = (df['skinTone'] >= 0.5).astype(int)

# Unique players by dark status
player_dark_counts = (
    df[['playerShort', 'skinTone', 'dark']]
    .drop_duplicates(subset=['playerShort'])
    .groupby('dark')['playerShort']
    .nunique()
)
print('\\nUnique players by dark status (threshold 0.5):')
print(player_dark_counts)

# Check consistency of skinTone per player
skin_variation = df.groupby('playerShort')['skinTone'].nunique()
print('\\nPlayers with >1 unique skinTone values:', (skin_variation > 1).sum())
print('Max unique skinTone values for a player:', skin_variation.max())

# Compute rate of red cards per game

df['red_rate'] = df['redCards'] / df['games']

# Group stats

group_stats = df.groupby('dark').agg(
    n=('red_rate','size'),
    total_red=('redCards','sum'),
    total_games=('games','sum'),
    mean_rate=('red_rate','mean')
).reset_index()
print('\nGroup stats (dark=1):')
print(group_stats)

# Poisson regression with offset(log(games))

# Add constant
X = sm.add_constant(df['skinTone'])
# Use Poisson GLM
poisson_model = sm.GLM(df['redCards'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
poisson_res = poisson_model.fit()
print('\nPoisson GLM (skinTone continuous)')
print(poisson_res.summary())

# Cluster-robust SE by playerShort
try:
    poisson_res_cluster = poisson_model.fit(cov_type='cluster', cov_kwds={'groups': df['playerShort']})
    print('\nPoisson GLM with cluster-robust SE by playerShort')
    print(poisson_res_cluster.summary())
except Exception as e:
    print('Cluster robust failed:', e)

# Binary dark vs light Poisson
X2 = sm.add_constant(df['dark'])
poisson_model2 = sm.GLM(df['redCards'], X2, family=sm.families.Poisson(), offset=np.log(df['games']))
poisson_res2 = poisson_model2.fit()
print('\nPoisson GLM (dark binary)')
print(poisson_res2.summary())

try:
    poisson_res2_cluster = poisson_model2.fit(cov_type='cluster', cov_kwds={'groups': df['playerShort']})
    print('\nPoisson GLM (dark binary) with cluster-robust SE by playerShort')
    print(poisson_res2_cluster.summary())
except Exception as e:
    print('Cluster robust failed:', e)

# Simple two-sample comparison of rates (per dyad)
from scipy import stats

light_rates = df.loc[df['dark']==0, 'red_rate']
dark_rates = df.loc[df['dark']==1, 'red_rate']

# T-test (unequal variances)

t_stat, p_val = stats.ttest_ind(dark_rates, light_rates, equal_var=False, nan_policy='omit')
print('\nT-test (red_rate per dyad, dark vs light): t=%.4f p=%.4g' % (t_stat, p_val))

# Mann-Whitney U test (nonparametric)

u_stat, p_u = stats.mannwhitneyu(dark_rates, light_rates, alternative='two-sided')
print('Mann-Whitney U test: U=%.4f p=%.4g' % (u_stat, p_u))

# Aggregated per player (total red cards / total games) to reduce dyad dependence

player_agg = df.groupby('playerShort').agg(
    skinTone=('skinTone','mean'),
    total_red=('redCards','sum'),
    total_games=('games','sum')
).reset_index()
player_agg['red_rate'] = player_agg['total_red'] / player_agg['total_games']
player_agg['dark'] = (player_agg['skinTone'] >= 0.5).astype(int)

print('\nPlayer-level aggregates:')
print(player_agg.groupby('dark').agg(n=('playerShort','size'), mean_rate=('red_rate','mean'), total_red=('total_red','sum'), total_games=('total_games','sum')))

# Poisson on player-level aggregated data
X3 = sm.add_constant(player_agg['skinTone'])
poisson_model3 = sm.GLM(player_agg['total_red'], X3, family=sm.families.Poisson(), offset=np.log(player_agg['total_games']))
poisson_res3 = poisson_model3.fit()
print('\nPoisson GLM on player-level aggregates (skinTone)')
print(poisson_res3.summary())

# dark binary
X4 = sm.add_constant(player_agg['dark'])
poisson_model4 = sm.GLM(player_agg['total_red'], X4, family=sm.families.Poisson(), offset=np.log(player_agg['total_games']))
poisson_res4 = poisson_model4.fit()
print('\nPoisson GLM on player-level aggregates (dark binary)')
print(poisson_res4.summary())
