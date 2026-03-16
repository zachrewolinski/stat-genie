import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest

DATA_PATH = "soccer.csv"

# Load dataset
# Use low_memory=False to avoid dtype inference warnings

df = pd.read_csv(DATA_PATH, low_memory=False)

# Compute mean skin tone
if {'rater1','rater2'}.issubset(df.columns):
    df['skin_mean'] = df[['rater1','rater2']].mean(axis=1)
else:
    raise ValueError("Missing rater columns")

# Filter to rows with skin data and valid games
base = df[df['skin_mean'].notna()].copy()
base = base[base['games'] > 0]

# Define light and dark groups based on scale (0-1)
# 0=very light, 0.25=light, 0.5=neutral, 0.75=dark, 1=very dark
light = base[base['skin_mean'] <= 0.25].copy()
dark = base[base['skin_mean'] >= 0.75].copy()

# Basic rates
light_red = light['redCards'].sum()
light_games = light['games'].sum()
light_rate = light_red / light_games

dark_red = dark['redCards'].sum()
dark_games = dark['games'].sum()
dark_rate = dark_red / dark_games

# Poisson regression for rate ratio dark vs light
if len(light) > 0 and len(dark) > 0:
    group = pd.concat([
        light.assign(dark=0),
        dark.assign(dark=1)
    ], ignore_index=True)

    model = sm.GLM(
        group['redCards'],
        sm.add_constant(group['dark']),
        family=sm.families.Poisson(),
        offset=np.log(group['games'])
    ).fit(cov_type='HC1')
    coef = model.params['dark']
    se = model.bse['dark']
    rr = float(np.exp(coef))
    # 95% CI for rate ratio
    ci_low = float(np.exp(coef - 1.96*se))
    ci_high = float(np.exp(coef + 1.96*se))
    p_value = float(model.pvalues['dark'])
else:
    rr = ci_low = ci_high = p_value = np.nan

# Continuous skin_mean model across full sample
cont = base.copy()
cont_model = sm.GLM(
    cont['redCards'],
    sm.add_constant(cont['skin_mean']),
    family=sm.families.Poisson(),
    offset=np.log(cont['games'])
).fit(cov_type='HC1')

cont_coef = cont_model.params['skin_mean']
cont_se = cont_model.bse['skin_mean']
# rate ratio for 0.1 increase in skin tone
rr_01 = float(np.exp(cont_coef * 0.1))
ci_01_low = float(np.exp((cont_coef - 1.96*cont_se) * 0.1))
ci_01_high = float(np.exp((cont_coef + 1.96*cont_se) * 0.1))
cont_p = float(cont_model.pvalues['skin_mean'])

results = {
    "n_rows": int(len(df)),
    "n_with_skin": int(len(base)),
    "light_n": int(len(light)),
    "dark_n": int(len(dark)),
    "light_red": float(light_red),
    "dark_red": float(dark_red),
    "light_games": float(light_games),
    "dark_games": float(dark_games),
    "light_rate": float(light_rate),
    "dark_rate": float(dark_rate),
    "rate_ratio_dark_vs_light": rr,
    "rr_ci_low": ci_low,
    "rr_ci_high": ci_high,
    "rr_p_value": p_value,
    "cont_rr_per_0.1": rr_01,
    "cont_ci_0.1_low": ci_01_low,
    "cont_ci_0.1_high": ci_01_high,
    "cont_p_value": cont_p,
}

print(json.dumps(results, indent=2))
