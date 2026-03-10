import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)

# Compute average skin tone (mean of raters)
# If one rater missing, use other. If both missing, result NaN.
skin = df[['feature18', 'feature19']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Keep rows with skin tone and games info
# feature9 is number of games in dyad
# feature16 is number of red cards in dyad

df = df[(~df['skin_tone'].isna()) & (~df['feature9'].isna()) & (~df['feature16'].isna())]

# Basic summary
n_rows = len(df)

# Create binary outcome: any red card in dyad
# also continuous count with offset

df['any_red'] = (df['feature16'] > 0).astype(int)

# Define light/dark categories: light <= 0.25, dark >= 0.75 (using 5-point scale normalized)
# This mirrors very light/light vs dark/very dark, excluding middle.
light_mask = df['skin_tone'] <= 0.25
_dark_mask = df['skin_tone'] >= 0.75

light = df[light_mask]
dark = df[_dark_mask]

# Compute red card rate per game for light/dark groups
light_rate = (light['feature16'].sum() / light['feature9'].sum()) if len(light) else np.nan

dark_rate = (dark['feature16'].sum() / dark['feature9'].sum()) if len(dark) else np.nan

# Rate ratio (dark / light)
rate_ratio = dark_rate / light_rate if (light_rate and dark_rate) else np.nan

# Poisson regression with offset for games
# Outcome: red card counts per dyad
# Predictor: skin_tone (continuous)
# offset: log(games)

# Avoid zero games (should be none per description, but safeguard)

df = df[df['feature9'] > 0]

# Add offset
# Use GLM Poisson with robust SEs (HC0)

poisson_model = smf.glm(
    formula='feature16 ~ skin_tone',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['feature9'])
).fit(cov_type='HC0')

# Extract coefficient
coef = poisson_model.params['skin_tone']
se = poisson_model.bse['skin_tone']
p_value = poisson_model.pvalues['skin_tone']

# Convert coef to rate ratio per 1.0 increase in skin_tone
rate_ratio_continuous = np.exp(coef)

# Also compute effect from light (0.25) to dark (0.75): delta=0.5
rate_ratio_dark_light = np.exp(coef * 0.5)

# Also logistic regression on any_red with games as offset? For logistic, we can include log(games) as covariate.
# Use GLM binomial with log(games) covariate to account for exposure.

# Add log games covariate

log_games = np.log(df['feature9'])

df = df.assign(log_games=log_games)

logit_model = smf.glm(
    formula='any_red ~ skin_tone + log_games',
    data=df,
    family=sm.families.Binomial()
).fit(cov_type='HC0')

logit_coef = logit_model.params['skin_tone']
logit_p = logit_model.pvalues['skin_tone']
logit_or = np.exp(logit_coef)

# Save results to a json-like text file for reference
results = {
    'n_rows': n_rows,
    'light_rows': int(light_mask.sum()),
    'dark_rows': int(_dark_mask.sum()),
    'light_rate': float(light_rate),
    'dark_rate': float(dark_rate),
    'rate_ratio_dark_light': float(rate_ratio),
    'poisson_coef': float(coef),
    'poisson_p': float(p_value),
    'poisson_rate_ratio_per_1': float(rate_ratio_continuous),
    'poisson_rate_ratio_0_25_to_0_75': float(rate_ratio_dark_light),
    'logit_coef': float(logit_coef),
    'logit_p': float(logit_p),
    'logit_or': float(logit_or),
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
