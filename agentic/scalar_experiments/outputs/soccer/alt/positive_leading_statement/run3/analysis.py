import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute skin tone per row (mean of raters)
# rater values already normalized 0-1; some missing
r1 = df['rater1']
r2 = df['rater2']
skin = pd.concat([r1, r2], axis=1).mean(axis=1, skipna=True)
df['skin_tone'] = skin

# Aggregate to player level to avoid dyad-level dependence
# Use playerShort as id
player_cols = ['playerShort']
agg = df.groupby('playerShort').agg(
    games=('games', 'sum'),
    redCards=('redCards', 'sum'),
    skin_tone=('skin_tone', 'mean'),
    yellowReds=('yellowReds', 'sum')
).reset_index()

# Keep players with skin tone and games > 0
agg = agg[(~agg['skin_tone'].isna()) & (agg['games'] > 0)]

# Add rate
agg['red_rate'] = agg['redCards'] / agg['games']

# Poisson regression with offset log(games)
# redCards ~ skin_tone
X = sm.add_constant(agg['skin_tone'])
poisson_model = sm.GLM(agg['redCards'], X, family=sm.families.Poisson(), offset=np.log(agg['games']))
poisson_res = poisson_model.fit(cov_type='HC3')

# Negative binomial (to check robustness)
nb_model = sm.GLM(agg['redCards'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(agg['games']))
nb_res = nb_model.fit(cov_type='HC3')

# Dark vs light groups based on quartiles/thresholds
light = agg[agg['skin_tone'] <= 0.25]
dark = agg[agg['skin_tone'] >= 0.75]

# Rate ratio using Poisson regression with binary indicator
# Only if both groups exist
rate_ratio = None
rate_ci = None
rate_p = None
n_light = len(light)
n_dark = len(dark)

if n_light > 0 and n_dark > 0:
    sub = agg[(agg['skin_tone'] <= 0.25) | (agg['skin_tone'] >= 0.75)].copy()
    sub['dark'] = (sub['skin_tone'] >= 0.75).astype(int)
    X2 = sm.add_constant(sub['dark'])
    pr = sm.GLM(sub['redCards'], X2, family=sm.families.Poisson(), offset=np.log(sub['games']))
    pr_res = pr.fit(cov_type='HC3')
    coef = pr_res.params['dark']
    se = pr_res.bse['dark']
    rate_ratio = float(np.exp(coef))
    # 95% CI
    rate_ci = (float(np.exp(coef - 1.96*se)), float(np.exp(coef + 1.96*se)))
    rate_p = float(pr_res.pvalues['dark'])

# Output summary
print('N players:', len(agg))
print('Total red cards:', agg['redCards'].sum())
print('Mean red rate per game:', agg['red_rate'].mean())
print('Poisson coef (skin_tone):', poisson_res.params['skin_tone'])
print('Poisson p-value:', poisson_res.pvalues['skin_tone'])
print('Poisson rate ratio per 1.0 increase:', np.exp(poisson_res.params['skin_tone']))
print('Poisson 95% CI:', np.exp(poisson_res.conf_int().loc['skin_tone']).tolist())
print('NB coef (skin_tone):', nb_res.params['skin_tone'])
print('NB p-value:', nb_res.pvalues['skin_tone'])
print('NB rate ratio per 1.0 increase:', np.exp(nb_res.params['skin_tone']))
print('NB 95% CI:', np.exp(nb_res.conf_int().loc['skin_tone']).tolist())
print('Light group n:', n_light, 'Dark group n:', n_dark)
print('Light red rate:', light['redCards'].sum() / light['games'].sum() if n_light > 0 else np.nan)
print('Dark red rate:', dark['redCards'].sum() / dark['games'].sum() if n_dark > 0 else np.nan)
print('Binary dark vs light rate ratio:', rate_ratio)
print('Binary 95% CI:', rate_ci)
print('Binary p-value:', rate_p)
