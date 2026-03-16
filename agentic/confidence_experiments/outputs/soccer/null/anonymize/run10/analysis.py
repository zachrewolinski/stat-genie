import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)

# Compute average skin tone (0 to 1)
skin_cols = ['feature18', 'feature19']

# Convert to numeric in case
for col in skin_cols + ['feature9', 'feature16']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['skin_avg'] = df[skin_cols].mean(axis=1, skipna=True)

# Filter: require skin tone and games > 0
work = df[(df['skin_avg'].notna()) & (df['feature9'] > 0)].copy()
work['games'] = work['feature9']
work['red_cards'] = work['feature16']

# Basic summary
summary = {
    'n_rows_total': int(df.shape[0]),
    'n_rows_with_skin': int(work.shape[0]),
    'skin_avg_min': float(work['skin_avg'].min()),
    'skin_avg_max': float(work['skin_avg'].max()),
    'skin_avg_mean': float(work['skin_avg'].mean()),
    'red_cards_total': float(work['red_cards'].sum()),
    'games_total': float(work['games'].sum()),
}

# Poisson regression with offset log(games)
X = sm.add_constant(work['skin_avg'])
offset = np.log(work['games'])
poisson_model = sm.GLM(work['red_cards'], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type='HC1')

# Negative binomial (GLM) with default alpha=1.0 (overdispersion)
# To estimate alpha, we can use statsmodels discrete model NB2, but GLM is okay for robustness check.
nb_model = sm.GLM(work['red_cards'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
nb_res = nb_model.fit(cov_type='HC1')

# Rate ratio for dark (1.0) vs light (0.0) per the continuous scale
coef = poisson_res.params['skin_avg']
rr_poisson = float(np.exp(coef))

coef_nb = nb_res.params['skin_avg']
rr_nb = float(np.exp(coef_nb))

# Group comparison for light vs dark (exclude middle)
light = work[work['skin_avg'] <= 0.25]
dark = work[work['skin_avg'] >= 0.75]

rate_light = light['red_cards'].sum() / light['games'].sum() if light.shape[0] else np.nan
rate_dark = dark['red_cards'].sum() / dark['games'].sum() if dark.shape[0] else np.nan
rate_ratio_group = float(rate_dark / rate_light) if rate_light and not np.isnan(rate_light) else np.nan

# Simple Poisson rate ratio test using approximate normal for log(rate ratio)
# var(log(rate_ratio)) ~ 1/red_dark + 1/red_light
red_dark = dark['red_cards'].sum()
red_light = light['red_cards'].sum()

if red_dark > 0 and red_light > 0:
    log_rr = np.log((red_dark / dark['games'].sum()) / (red_light / light['games'].sum()))
    se_log_rr = np.sqrt(1 / red_dark + 1 / red_light)
    z = log_rr / se_log_rr
    p_value_group = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_value_group = np.nan

results = {
    'summary': summary,
    'poisson': {
        'coef_skin': float(coef),
        'se_skin': float(poisson_res.bse['skin_avg']),
        'p_value_skin': float(poisson_res.pvalues['skin_avg']),
        'rate_ratio_1_vs_0': rr_poisson,
    },
    'neg_binom': {
        'coef_skin': float(coef_nb),
        'se_skin': float(nb_res.bse['skin_avg']),
        'p_value_skin': float(nb_res.pvalues['skin_avg']),
        'rate_ratio_1_vs_0': rr_nb,
    },
    'group_rates': {
        'n_light_rows': int(light.shape[0]),
        'n_dark_rows': int(dark.shape[0]),
        'red_light': float(red_light),
        'games_light': float(light['games'].sum()),
        'rate_light': float(rate_light) if not np.isnan(rate_light) else None,
        'red_dark': float(red_dark),
        'games_dark': float(dark['games'].sum()),
        'rate_dark': float(rate_dark) if not np.isnan(rate_dark) else None,
        'rate_ratio_dark_vs_light': rate_ratio_group if not np.isnan(rate_ratio_group) else None,
        'z_score': float(z) if not np.isnan(z) else None,
        'p_value': float(p_value_group) if not np.isnan(p_value_group) else None,
    },
}

print(json.dumps(results, indent=2))
