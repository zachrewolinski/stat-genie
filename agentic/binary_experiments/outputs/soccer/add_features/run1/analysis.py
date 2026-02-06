import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Compute skin tone (mean of two raters), keep players with ratings
_df['skin_tone'] = _df[['rater1', 'rater2']].mean(axis=1)
_df = _df.dropna(subset=['skin_tone', 'games', 'redCards'])

# Aggregate to player level to answer player-level question
player = (
    _df.groupby('playerShort', as_index=False)
    .agg(
        games=('games', 'sum'),
        redCards=('redCards', 'sum'),
        skin_tone=('skin_tone', 'mean'),
    )
)

# Define dark vs light using mid-point of 0.5 on normalized 0-1 scale
player['dark'] = (player['skin_tone'] > 0.5).astype(int)

# Avoid zero exposure
player = player[player['games'] > 0].copy()

# Rates per 100 games
rate_dark = (player.loc[player['dark'] == 1, 'redCards'].sum() /
             player.loc[player['dark'] == 1, 'games'].sum()) * 100
rate_light = (player.loc[player['dark'] == 0, 'redCards'].sum() /
              player.loc[player['dark'] == 0, 'games'].sum()) * 100

# Poisson regression with exposure offset
X = sm.add_constant(player['dark'])
offset = np.log(player['games'])
model = sm.GLM(player['redCards'], X, family=sm.families.Poisson(), offset=offset)
result = model.fit()

rate_ratio = np.exp(result.params['dark'])
p_value = result.pvalues['dark']

# Save a small summary for reference
summary = {
    'n_players': int(len(player)),
    'n_dark_players': int(player['dark'].sum()),
    'n_light_players': int((player['dark'] == 0).sum()),
    'rate_dark_per_100_games': float(rate_dark),
    'rate_light_per_100_games': float(rate_light),
    'rate_ratio_dark_vs_light': float(rate_ratio),
    'p_value': float(p_value),
}

pd.Series(summary).to_csv('analysis_summary.csv')

# Write conclusion
more_likely = (rate_ratio > 1.0) and (p_value < 0.05)
with open('conclusion.txt', 'w') as f:
    f.write('Yes\n' if more_likely else 'No\n')
    if more_likely:
        f.write(
            f"Dark-skinned players show a higher red-card rate than light-skinned players "
            f"(rate ratio {rate_ratio:.2f}, p={p_value:.3g}; "
            f"{rate_dark:.2f} vs {rate_light:.2f} per 100 games).\n"
        )
    else:
        f.write(
            f"Red-card rates are not higher for dark-skinned players based on this data "
            f"(rate ratio {rate_ratio:.2f}, p={p_value:.3g}; "
            f"{rate_dark:.2f} vs {rate_light:.2f} per 100 games).\n"
        )
