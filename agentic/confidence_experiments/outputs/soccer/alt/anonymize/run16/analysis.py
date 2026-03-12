import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# Load data
file_path = 'soccer.csv'
df = pd.read_csv(file_path)

# Compute mean skin tone across two raters
skin_cols = ['feature18', 'feature19']
df['skin_mean'] = df[skin_cols].mean(axis=1, skipna=True)

# Player-level aggregation
player_df = df.groupby('feature1', as_index=False).agg(
    skin_mean=('skin_mean', 'mean'),
    games=('feature9', 'sum'),
    red_cards=('feature16', 'sum')
)

# Remove players without skin ratings or games
player_df = player_df.dropna(subset=['skin_mean'])
player_df = player_df[player_df['games'] > 0]

# Poisson regression: red_cards ~ skin_mean with offset log(games)
# Use robust (HC1) standard errors for potential overdispersion
X = sm.add_constant(player_df['skin_mean'])
model = sm.GLM(player_df['red_cards'], X, family=sm.families.Poisson(), offset=np.log(player_df['games']))
result = model.fit(cov_type='HC1')

# Extract effect (rate ratio for full-scale skin_mean 0->1)
coef = result.params['skin_mean']
se = result.bse['skin_mean']
rr = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)
p_value = result.pvalues['skin_mean']

# Group comparison: dark vs light (extremes)
# Define light <= 0.25, dark >= 0.75 (scale 0-1)
light = player_df[player_df['skin_mean'] <= 0.25]
dark = player_df[player_df['skin_mean'] >= 0.75]

# Compute rate per game and rate ratio
light_red = light['red_cards'].sum()
light_games = light['games'].sum()
dark_red = dark['red_cards'].sum()
dark_games = dark['games'].sum()

# Avoid divide-by-zero
if light_red > 0 and dark_red > 0:
    light_rate = light_red / light_games
    dark_rate = dark_red / dark_games
    rate_ratio = dark_rate / light_rate
    # Standard error for log rate ratio (Poisson counts)
    se_log_rr = np.sqrt(1 / dark_red + 1 / light_red)
    z = np.log(rate_ratio) / se_log_rr
    p_value_group = 2 * (1 - norm.cdf(abs(z)))
    ci_low_rr = np.exp(np.log(rate_ratio) - 1.96 * se_log_rr)
    ci_high_rr = np.exp(np.log(rate_ratio) + 1.96 * se_log_rr)
else:
    light_rate = dark_rate = rate_ratio = np.nan
    p_value_group = np.nan
    ci_low_rr = ci_high_rr = np.nan

# Assemble summary
summary = {
    'n_players_total': int(player_df.shape[0]),
    'n_players_light': int(light.shape[0]),
    'n_players_dark': int(dark.shape[0]),
    'poisson_coef': float(coef),
    'poisson_se': float(se),
    'poisson_rr': float(rr),
    'poisson_ci_low': float(ci_low),
    'poisson_ci_high': float(ci_high),
    'poisson_p_value': float(p_value),
    'light_rate_per_game': float(light_rate) if not np.isnan(light_rate) else None,
    'dark_rate_per_game': float(dark_rate) if not np.isnan(dark_rate) else None,
    'rate_ratio_dark_vs_light': float(rate_ratio) if not np.isnan(rate_ratio) else None,
    'rr_ci_low': float(ci_low_rr) if not np.isnan(ci_low_rr) else None,
    'rr_ci_high': float(ci_high_rr) if not np.isnan(ci_high_rr) else None,
    'p_value_group': float(p_value_group) if not np.isnan(p_value_group) else None
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
