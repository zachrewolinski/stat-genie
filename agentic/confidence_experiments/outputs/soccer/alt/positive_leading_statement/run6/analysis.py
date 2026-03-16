import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest

# Load data
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

# Compute average skin tone where available
# rater1 and rater2 in [0,1] 5-point scale
skin = df[['rater1', 'rater2']].mean(axis=1)
df = df.assign(skin_tone=skin)

# Keep rows with games >=1 and skin_tone not missing and redCards not missing
analysis_df = df.dropna(subset=['skin_tone', 'games', 'redCards'])
analysis_df = analysis_df[analysis_df['games'] > 0]

# Create dark vs light groups (top/bottom two categories)
# Categories likely 0, 0.25, 0.5, 0.75, 1
analysis_df = analysis_df.assign(
    light=analysis_df['skin_tone'] <= 0.25,
    dark=analysis_df['skin_tone'] >= 0.75,
)

# Restrict group comparison to light/dark only
ld_df = analysis_df[analysis_df['light'] | analysis_df['dark']].copy()
ld_df['dark_indicator'] = ld_df['dark'].astype(int)

# Poisson regression: redCards ~ dark_indicator with offset log(games)
# Use robust covariance
ld_df['log_games'] = np.log(ld_df['games'])
poisson_dark = sm.GLM(
    ld_df['redCards'],
    sm.add_constant(ld_df['dark_indicator']),
    family=sm.families.Poisson(),
    offset=ld_df['log_games'],
)
poisson_dark_res = poisson_dark.fit(cov_type='HC0')

# Poisson regression: redCards ~ skin_tone (continuous) with offset
analysis_df['log_games'] = np.log(analysis_df['games'])
poisson_cont = sm.GLM(
    analysis_df['redCards'],
    sm.add_constant(analysis_df['skin_tone']),
    family=sm.families.Poisson(),
    offset=analysis_df['log_games'],
)
poisson_cont_res = poisson_cont.fit(cov_type='HC0')

# Negative binomial (to check robustness)
nb_cont = sm.GLM(
    analysis_df['redCards'],
    sm.add_constant(analysis_df['skin_tone']),
    family=sm.families.NegativeBinomial(),
    offset=analysis_df['log_games'],
)
nb_cont_res = nb_cont.fit(cov_type='HC0')

# Aggregate rates for dark vs light
agg = ld_df.groupby('dark_indicator').agg(
    red_cards=('redCards', 'sum'),
    games=('games', 'sum'),
    n=('redCards', 'size')
).reset_index()

# Compute rate per game and rate ratio (dark vs light)
if agg.shape[0] == 2:
    rate_light = agg.loc[agg['dark_indicator'] == 0, 'red_cards'].iloc[0] / agg.loc[agg['dark_indicator'] == 0, 'games'].iloc[0]
    rate_dark = agg.loc[agg['dark_indicator'] == 1, 'red_cards'].iloc[0] / agg.loc[agg['dark_indicator'] == 1, 'games'].iloc[0]
    rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan
else:
    rate_light = rate_dark = rate_ratio = np.nan

# Extract results
coef_dark = poisson_dark_res.params['dark_indicator']
se_dark = poisson_dark_res.bse['dark_indicator']
p_dark = poisson_dark_res.pvalues['dark_indicator']
irr_dark = float(np.exp(coef_dark))

coef_skin = poisson_cont_res.params['skin_tone']
se_skin = poisson_cont_res.bse['skin_tone']
p_skin = poisson_cont_res.pvalues['skin_tone']
irr_skin = float(np.exp(coef_skin))
# Rate ratio for dark (0.75) vs light (0.25) in continuous model
irr_dark_vs_light = float(np.exp(coef_skin * (0.75 - 0.25)))

coef_skin_nb = nb_cont_res.params['skin_tone']
se_skin_nb = nb_cont_res.bse['skin_tone']
p_skin_nb = nb_cont_res.pvalues['skin_tone']
irr_skin_nb = float(np.exp(coef_skin_nb))

# Sample sizes
n_total = analysis_df.shape[0]

# Compose results
results = {
    'n_total': int(n_total),
    'n_light_dark': int(ld_df.shape[0]),
    'poisson_dark': {
        'coef': float(coef_dark),
        'se': float(se_dark),
        'p_value': float(p_dark),
        'irr_dark_vs_light': irr_dark,
    },
    'poisson_cont': {
        'coef': float(coef_skin),
        'se': float(se_skin),
        'p_value': float(p_skin),
        'irr_per_unit': irr_skin,
        'irr_dark_vs_light': irr_dark_vs_light,
    },
    'nb_cont': {
        'coef': float(coef_skin_nb),
        'se': float(se_skin_nb),
        'p_value': float(p_skin_nb),
        'irr_per_unit': irr_skin_nb,
    },
    'rates': {
        'rate_light': float(rate_light),
        'rate_dark': float(rate_dark),
        'rate_ratio_dark_light': float(rate_ratio),
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
