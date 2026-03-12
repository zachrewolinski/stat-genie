import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Map columns based on metadata descriptions
skin1 = df['rater1']
skin2 = df['nExp']  # rater2 (normalized 0-1) based on description
red_cards = df['yellowCards']  # number of red cards received
exposure_games = df['redCards']  # number of games in dyad

# Compute mean skin tone
skin_tone = pd.concat([skin1, skin2], axis=1).mean(axis=1)

# Build analysis dataframe
analysis_df = pd.DataFrame({
    'skin_tone': skin_tone,
    'red_cards': red_cards,
    'games': exposure_games
})

# Drop missing or invalid rows
analysis_df = analysis_df.dropna()
analysis_df = analysis_df[analysis_df['games'] > 0]

# Continuous Poisson regression with offset
X_cont = sm.add_constant(analysis_df['skin_tone'])
model_cont = sm.GLM(
    analysis_df['red_cards'],
    X_cont,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['games'])
).fit()

# Dark vs light groups
light_mask = analysis_df['skin_tone'] <= 0.25
dark_mask = analysis_df['skin_tone'] >= 0.75

light = analysis_df[light_mask]
dark = analysis_df[dark_mask]

# Group rates
light_red = light['red_cards'].sum()
dark_red = dark['red_cards'].sum()
light_games = light['games'].sum()
dark_games = dark['games'].sum()

light_rate = light_red / light_games if light_games > 0 else np.nan
dark_rate = dark_red / dark_games if dark_games > 0 else np.nan
rate_ratio = (dark_rate / light_rate) if (light_rate and light_rate > 0) else np.nan

# Poisson regression for dark vs light only
subset = analysis_df[light_mask | dark_mask].copy()
subset['dark'] = (subset['skin_tone'] >= 0.75).astype(int)
X_bin = sm.add_constant(subset['dark'])
model_bin = sm.GLM(
    subset['red_cards'],
    X_bin,
    family=sm.families.Poisson(),
    offset=np.log(subset['games'])
).fit()

# Summaries
results = {
    'n_total': len(analysis_df),
    'n_light': len(light),
    'n_dark': len(dark),
    'light_red_cards': int(light_red),
    'dark_red_cards': int(dark_red),
    'light_games': float(light_games),
    'dark_games': float(dark_games),
    'light_rate': light_rate,
    'dark_rate': dark_rate,
    'rate_ratio_dark_vs_light': rate_ratio,
    'poisson_cont_coef': model_cont.params['skin_tone'],
    'poisson_cont_pvalue': model_cont.pvalues['skin_tone'],
    'poisson_bin_coef': model_bin.params['dark'],
    'poisson_bin_pvalue': model_bin.pvalues['dark'],
}

print(results)
