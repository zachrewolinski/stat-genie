import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('soccer.csv')

# Create skin tone metric (mean of raters)
_df['skin_tone'] = _df[['rater1', 'rater2']].mean(axis=1, skipna=True)

# Filter rows with skin tone and games info
_df = _df.dropna(subset=['skin_tone', 'games', 'redCards'])

# Basic counts
print('rows:', len(_df))
print('unique players:', _df['playerShort'].nunique())

# Dyad-level Poisson regression: redCards ~ skin_tone with log(games) offset
# Add small epsilon to games to avoid log(0), though games min should be 1
_df['log_games'] = np.log(_df['games'].clip(lower=1))

X = sm.add_constant(_df['skin_tone'])
poisson_model = sm.GLM(_df['redCards'], X, family=sm.families.Poisson(), offset=_df['log_games'])
poisson_res = poisson_model.fit()
print('\nDyad-level Poisson with offset log(games):')
print(poisson_res.summary().tables[1])

# Compute incidence rate ratio for skin_tone
coef = poisson_res.params['skin_tone']
se = poisson_res.bse['skin_tone']
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print(f"IRR per 1.0 skin_tone: {irr:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f})")

# Player-level aggregation
player = (
    _df.groupby('playerShort', as_index=False)
       .agg({
           'skin_tone': 'mean',
           'redCards': 'sum',
           'games': 'sum'
       })
)
player['red_rate'] = player['redCards'] / player['games']

# Correlation test between skin_tone and red_rate
corr, p_corr = stats.pearsonr(player['skin_tone'], player['red_rate'])
print('\nPlayer-level correlation (skin_tone vs red_rate):')
print('r:', corr, 'p:', p_corr)

# Player-level Poisson regression (offset log(games))
player['log_games'] = np.log(player['games'].clip(lower=1))
X_p = sm.add_constant(player['skin_tone'])
poisson_p = sm.GLM(player['redCards'], X_p, family=sm.families.Poisson(), offset=player['log_games'])
poisson_p_res = poisson_p.fit()
print('\nPlayer-level Poisson with offset log(games):')
print(poisson_p_res.summary().tables[1])
coef_p = poisson_p_res.params['skin_tone']
se_p = poisson_p_res.bse['skin_tone']
irr_p = np.exp(coef_p)
ci_low_p = np.exp(coef_p - 1.96*se_p)
ci_high_p = np.exp(coef_p + 1.96*se_p)
print(f"IRR per 1.0 skin_tone: {irr_p:.3f} (95% CI {ci_low_p:.3f}, {ci_high_p:.3f})")

# Group comparison: light vs dark
# Define light <=0.25, dark >=0.75
light = player[player['skin_tone'] <= 0.25]
dark = player[player['skin_tone'] >= 0.75]

print('\nLight vs Dark (player-level, extreme groups):')
print('light n:', len(light), 'dark n:', len(dark))
print('mean red_rate light:', light['red_rate'].mean(), 'dark:', dark['red_rate'].mean())

# t-test for red_rate between light and dark (Welch)
if len(light) > 1 and len(dark) > 1:
    t_stat, p_val = stats.ttest_ind(light['red_rate'], dark['red_rate'], equal_var=False)
    print('t-test red_rate light vs dark: t=', t_stat, 'p=', p_val)

# Rate ratio for total red cards per games
light_total_rate = light['redCards'].sum() / light['games'].sum()
dark_total_rate = dark['redCards'].sum() / dark['games'].sum()
print('total rate light:', light_total_rate, 'dark:', dark_total_rate, 'ratio dark/light:', dark_total_rate / light_total_rate if light_total_rate>0 else np.nan)

