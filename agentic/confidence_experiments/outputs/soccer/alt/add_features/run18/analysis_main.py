import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = Path('soccer.csv')

df = pd.read_csv(DATA_PATH)

# compute mean skin tone across raters
skin = df[['rater1', 'rater2']].mean(axis=1)
df = df.assign(skin=skin)

# drop rows without skin rating
_df = df.dropna(subset=['skin']).copy()

# aggregate to player level (reduce dyad dependence)
# use playerShort as id
agg = (
    _df.groupby('playerShort', as_index=False)
    .agg(
        skin=('skin', 'mean'),
        games=('games', 'sum'),
        redCards=('redCards', 'sum')
    )
)

# filter to players with at least 1 game (should already)
agg = agg[agg['games'] > 0].copy()

# binary groups for light vs dark (exclude midtones)
light_mask = agg['skin'] <= 0.25
Dark_mask = agg['skin'] >= 0.75

light = agg[light_mask].copy()
dark = agg[Dark_mask].copy()

# compute rates per 100 games
light_rate = (light['redCards'].sum() / light['games'].sum()) * 100 if len(light) else np.nan
dark_rate = (dark['redCards'].sum() / dark['games'].sum()) * 100 if len(dark) else np.nan

# Poisson regression with offset for exposure
# continuous skin tone
agg['log_games'] = np.log(agg['games'])
X = sm.add_constant(agg['skin'])
poisson_model = sm.GLM(agg['redCards'], X, family=sm.families.Poisson(), offset=agg['log_games'])
poisson_res = poisson_model.fit(cov_type='HC1')

# effect per 0.25 increase (one skin-step on 5-point scale)
coef = poisson_res.params['skin']
se = poisson_res.bse['skin']

# compute IRR for 0.25 increase
irr_step = math.exp(coef * 0.25)
# 95% CI
z = 1.96
ci_low = math.exp((coef - z * se) * 0.25)
ci_high = math.exp((coef + z * se) * 0.25)

p_value = poisson_res.pvalues['skin']

# Poisson regression for dark vs light (binary) on subset
# only if both groups exist
binary_results = None
if len(light) > 0 and len(dark) > 0:
    sub = pd.concat([light.assign(group=0), dark.assign(group=1)], ignore_index=True)
    sub['log_games'] = np.log(sub['games'])
    Xb = sm.add_constant(sub['group'])
    model_b = sm.GLM(sub['redCards'], Xb, family=sm.families.Poisson(), offset=sub['log_games'])
    res_b = model_b.fit(cov_type='HC1')
    coef_b = res_b.params['group']
    se_b = res_b.bse['group']
    p_b = res_b.pvalues['group']
    irr_b = math.exp(coef_b)
    ci_b_low = math.exp(coef_b - z * se_b)
    ci_b_high = math.exp(coef_b + z * se_b)
    binary_results = {
        'irr': irr_b,
        'ci_low': ci_b_low,
        'ci_high': ci_b_high,
        'p': p_b,
        'n_light': len(light),
        'n_dark': len(dark)
    }

# save summary for inspection
summary = {
    'n_rows': len(df),
    'n_skin': len(_df),
    'n_players': len(agg),
    'n_light': len(light),
    'n_dark': len(dark),
    'light_rate_per100': light_rate,
    'dark_rate_per100': dark_rate,
    'poisson_skin_coef': coef,
    'poisson_skin_se': se,
    'poisson_skin_p': p_value,
    'irr_step_0_25': irr_step,
    'irr_step_ci_low': ci_low,
    'irr_step_ci_high': ci_high,
    'binary': binary_results,
}

print(json.dumps(summary, indent=2))
