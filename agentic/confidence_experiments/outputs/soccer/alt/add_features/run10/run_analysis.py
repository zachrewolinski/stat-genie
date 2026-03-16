import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

# Compute mean skin tone from raters (normalized 0-1)
# Use available raters; if both missing, skin tone is NaN
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Filter to rows with skin tone and positive games
analysis_df = df[(df['skin_tone'].notna()) & (df['games'] > 0)].copy()

# Create dark/light groups based on 5-point scale normalized to 0..1
# 0.00 = very light, 0.25 = light, 0.50 = medium, 0.75 = dark, 1.00 = very dark
analysis_df['skin_group'] = pd.cut(
    analysis_df['skin_tone'],
    bins=[-0.01, 0.125, 0.375, 0.625, 0.875, 1.01],
    labels=['very_light', 'light', 'medium', 'dark', 'very_dark']
)

# Aggregate by player for descriptive rates
player_agg = analysis_df.groupby('playerShort', as_index=False).agg(
    skin_tone=('skin_tone', 'mean'),
    games=('games', 'sum'),
    redCards=('redCards', 'sum')
)

# Assign player-level group (use mean skin tone across dyads)
player_agg['skin_group'] = pd.cut(
    player_agg['skin_tone'],
    bins=[-0.01, 0.125, 0.375, 0.625, 0.875, 1.01],
    labels=['very_light', 'light', 'medium', 'dark', 'very_dark']
)

# Define light vs dark groups for comparison (light = very_light or light; dark = dark or very_dark)
player_agg['light_vs_dark'] = np.where(
    player_agg['skin_group'].isin(['very_light', 'light']), 'light',
    np.where(player_agg['skin_group'].isin(['dark', 'very_dark']), 'dark', np.nan)
)

player_ld = player_agg[player_agg['light_vs_dark'].isin(['light', 'dark'])].copy()

# Compute rates per game by group
rate_by_group = player_ld.groupby('light_vs_dark').apply(
    lambda g: pd.Series({
        'players': g.shape[0],
        'total_games': g['games'].sum(),
        'total_red': g['redCards'].sum(),
        'red_per_game': g['redCards'].sum() / g['games'].sum() if g['games'].sum() > 0 else np.nan
    })
).reset_index()

# Poisson regression at dyad-level with offset for games
# Model 1: continuous skin tone
analysis_df['log_games'] = np.log(analysis_df['games'])
X_cont = sm.add_constant(analysis_df['skin_tone'])
model_cont = sm.GLM(analysis_df['redCards'], X_cont, family=sm.families.Poisson(), offset=analysis_df['log_games'])
res_cont = model_cont.fit(cov_type='cluster', cov_kwds={'groups': analysis_df['playerShort']})

# Model 2: dark vs light only, dyad-level, with offset
analysis_ld = analysis_df.copy()
analysis_ld['light_vs_dark'] = np.where(
    analysis_ld['skin_group'].isin(['very_light', 'light']), 'light',
    np.where(analysis_ld['skin_group'].isin(['dark', 'very_dark']), 'dark', np.nan)
)
analysis_ld = analysis_ld[analysis_ld['light_vs_dark'].isin(['light', 'dark'])].copy()
analysis_ld['dark'] = (analysis_ld['light_vs_dark'] == 'dark').astype(int)

X_dark = sm.add_constant(analysis_ld['dark'])
model_dark = sm.GLM(analysis_ld['redCards'], X_dark, family=sm.families.Poisson(), offset=np.log(analysis_ld['games']))
res_dark = model_dark.fit(cov_type='cluster', cov_kwds={'groups': analysis_ld['playerShort']})

# Extract results
cont_coef = res_cont.params['skin_tone']
cont_se = res_cont.bse['skin_tone']
cont_p = res_cont.pvalues['skin_tone']
cont_irr = float(np.exp(cont_coef))

# Dark vs light IRR
if 'dark' in res_dark.params.index:
    dark_coef = res_dark.params['dark']
    dark_se = res_dark.bse['dark']
    dark_p = res_dark.pvalues['dark']
    dark_irr = float(np.exp(dark_coef))
else:
    dark_coef = dark_se = dark_p = dark_irr = np.nan

# Descriptive rate ratio
rate_map = {row['light_vs_dark']: row['red_per_game'] for _, row in rate_by_group.iterrows()}
rate_ratio = rate_map.get('dark', np.nan) / rate_map.get('light', np.nan)

# Save summary stats to a JSON for manual inspection
summary = {
    'rows_total': int(df.shape[0]),
    'rows_with_skin': int(analysis_df.shape[0]),
    'players_with_skin': int(player_agg.shape[0]),
    'player_group_rates': rate_by_group.to_dict(orient='records'),
    'rate_ratio_dark_over_light': float(rate_ratio),
    'poisson_continuous': {
        'coef': float(cont_coef),
        'se': float(cont_se),
        'p_value': float(cont_p),
        'irr': float(cont_irr)
    },
    'poisson_dark_vs_light': {
        'coef': float(dark_coef),
        'se': float(dark_se),
        'p_value': float(dark_p),
        'irr': float(dark_irr)
    },
    'n_dyads_dark_vs_light': int(analysis_ld.shape[0])
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
