import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.rates import test_poisson_2indep

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Skin tone: average of two raters (0=very light, 1=very dark)
skin_raters = df[['rater1', 'nExp']]
skin_mean = skin_raters.mean(axis=1, skipna=True)
mask_skin = skin_raters.notna().any(axis=1)

# Outcome and exposure
red_cards = df['yellowCards']  # per metadata, this is red cards in the dyad
matches = df['redCards']       # per metadata, this is number of games in the dyad

# Filter valid rows
mask = mask_skin & red_cards.notna() & matches.notna() & (matches > 0)
use = df.loc[mask].copy()
use['skin_mean'] = skin_mean[mask]

# Group definition for dark vs light
light_mask = use['skin_mean'] <= 0.25
# dark: darker categories
# (0.75 or 1.0 are dark/very dark; intermediate values treated as middle and excluded)
dark_mask = use['skin_mean'] >= 0.75

light = use.loc[light_mask]
dark = use.loc[dark_mask]

# Aggregate rates
light_red = light['yellowCards'].sum()
light_games = light['redCards'].sum()
dark_red = dark['yellowCards'].sum()
dark_games = dark['redCards'].sum()

light_rate = light_red / light_games if light_games > 0 else np.nan
dark_rate = dark_red / dark_games if dark_games > 0 else np.nan

# Poisson test for rate difference
rate_test = test_poisson_2indep(
    count1=dark_red,
    exposure1=dark_games,
    count2=light_red,
    exposure2=light_games,
    method='wald',
    alternative='two-sided'
)

rate_ratio = (dark_red / dark_games) / (light_red / light_games)

# Poisson regression with offset (continuous skin tone)
X = sm.add_constant(use['skin_mean'])
model = sm.GLM(
    use['yellowCards'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(use['redCards'])
)
res = model.fit(cov_type='HC3')
coef = res.params['skin_mean']
coef_se = res.bse['skin_mean']
coef_p = res.pvalues['skin_mean']

# Effect from light (0.25) to dark (0.75): delta=0.5
rr_05 = float(np.exp(coef * 0.5))

# Predicted rates per game at 0.25 and 0.75
intercept = res.params['const']
rate_light_pred = float(np.exp(intercept + coef * 0.25))
rate_dark_pred = float(np.exp(intercept + coef * 0.75))

# Summaries
summary = {
    'n_rows': int(use.shape[0]),
    'light_rows': int(light.shape[0]),
    'dark_rows': int(dark.shape[0]),
    'light_red_cards_total': int(light_red),
    'light_games_total': int(light_games),
    'dark_red_cards_total': int(dark_red),
    'dark_games_total': int(dark_games),
    'light_rate': float(light_rate),
    'dark_rate': float(dark_rate),
    'rate_ratio_dark_vs_light': float(rate_ratio),
    'rate_test_pvalue': float(rate_test.pvalue),
    'poisson_coef_skin': float(coef),
    'poisson_coef_se': float(coef_se),
    'poisson_coef_pvalue': float(coef_p),
    'rr_from_0.25_to_0.75': rr_05,
    'pred_rate_light_0.25': rate_light_pred,
    'pred_rate_dark_0.75': rate_dark_pred,
}

print(json.dumps(summary, indent=2))
