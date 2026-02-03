import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Compute average skin tone from two raters
_df['skin_avg'] = _df[['rater1', 'rater2']].mean(axis=1)

# Keep rows with skin tone and positive games
_df = _df.dropna(subset=['skin_avg', 'games', 'redCards'])
_df = _df[_df['games'] > 0]

# Define light/dark groups based on the 5-point scale mapped to 0..1
# light: very light/light (<= 0.25), dark: dark/very dark (>= 0.75)
_df['skin_group'] = np.where(
    _df['skin_avg'] <= 0.25, 'light',
    np.where(_df['skin_avg'] >= 0.75, 'dark', 'mid')
)

# Aggregate rates by group
_group_stats = (
    _df.groupby('skin_group', as_index=False)
       .agg(red_cards=('redCards', 'sum'), games=('games', 'sum'))
)
_group_stats['red_per_game'] = _group_stats['red_cards'] / _group_stats['games']

# Rate ratio dark vs light
_light = _group_stats[_group_stats['skin_group'] == 'light'].iloc[0]
_dark = _group_stats[_group_stats['skin_group'] == 'dark'].iloc[0]
rate_ratio = (_dark['red_per_game'] / _light['red_per_game']) if _light['red_per_game'] > 0 else np.nan

# Poisson regression with exposure (games) and continuous skin tone
# redCards ~ skin_avg + offset(log(games))
X = sm.add_constant(_df['skin_avg'])
model = sm.GLM(_df['redCards'], X, family=sm.families.Poisson(), offset=np.log(_df['games']))
result = model.fit(cov_type='HC0')
coef = result.params['skin_avg']
se = result.bse['skin_avg']
p_value = result.pvalues['skin_avg']

# Convert coefficient to rate ratio per 1.0 increase in skin_avg
rate_ratio_cont = np.exp(coef)

# Save key results to a small table for reference
summary = {
    'n_rows': len(_df),
    'group_stats': _group_stats.to_dict(orient='list'),
    'rate_ratio_dark_vs_light': float(rate_ratio),
    'poisson_coef_skin_avg': float(coef),
    'poisson_se_skin_avg': float(se),
    'poisson_p_value_skin_avg': float(p_value),
    'poisson_rate_ratio_per_1.0_skin': float(rate_ratio_cont),
}

print(summary)
