import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def irr_ci(res, param):
    coef = res.params[param]
    se = res.bse[param]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    return np.exp(coef), np.exp(ci_low), np.exp(ci_high)


df = pd.read_csv('soccer.csv')

# Compute mean skin rating from available raters
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)

df = df.assign(skin=skin)

# Basic filters
df = df[(df['games'] > 0) & df['skin'].notna() & df['redCards'].notna()]

# Rates
df['red_per_game'] = df['redCards'] / df['games']

# Strict dark vs light split, excluding exact midpoint (0.5)
df_strict = df[df['skin'] != 0.5].copy()
df_strict['dark'] = (df_strict['skin'] > 0.5).astype(int)

# Group stats
light = df_strict[df_strict['dark'] == 0]
dark = df_strict[df_strict['dark'] == 1]

summary = {
    'n_total': len(df),
    'n_strict': len(df_strict),
    'n_light': len(light),
    'n_dark': len(dark),
    'mean_red_per_game_light': light['red_per_game'].mean(),
    'mean_red_per_game_dark': dark['red_per_game'].mean(),
    'mean_red_per_game_all': df['red_per_game'].mean(),
}

# T-test for rate difference
if len(light) > 1 and len(dark) > 1:
    t_stat, t_p = stats.ttest_ind(light['red_per_game'], dark['red_per_game'], equal_var=False)
else:
    t_stat, t_p = np.nan, np.nan

summary.update({'t_stat': t_stat, 't_p': t_p})

# Poisson regression with offset for exposure (games)
# Continuous skin
poisson_skin = smf.glm(
    'redCards ~ skin',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['games'])
).fit(cov_type='HC1')

# Binary dark indicator
poisson_dark = smf.glm(
    'redCards ~ dark',
    data=df_strict,
    family=sm.families.Poisson(),
    offset=np.log(df_strict['games'])
).fit(cov_type='HC1')

# Logistic regression on any red card
# Add log games as exposure proxy

df['any_red'] = (df['redCards'] > 0).astype(int)
logit = smf.logit('any_red ~ skin + np.log(games)', data=df).fit(disp=False)

# Extract IRR/OR and CIs
irr_skin = irr_ci(poisson_skin, 'skin')
irr_dark = irr_ci(poisson_dark, 'dark')

or_skin = np.exp(logit.params['skin'])
or_skin_ci = np.exp(logit.conf_int().loc['skin'])

out = {
    'summary': summary,
    'poisson_skin': {
        'coef': poisson_skin.params['skin'],
        'p': poisson_skin.pvalues['skin'],
        'irr': irr_skin[0],
        'irr_ci_low': irr_skin[1],
        'irr_ci_high': irr_skin[2],
    },
    'poisson_dark': {
        'coef': poisson_dark.params['dark'],
        'p': poisson_dark.pvalues['dark'],
        'irr': irr_dark[0],
        'irr_ci_low': irr_dark[1],
        'irr_ci_high': irr_dark[2],
    },
    'logit_skin': {
        'coef': logit.params['skin'],
        'p': logit.pvalues['skin'],
        'or': or_skin,
        'or_ci_low': or_skin_ci[0],
        'or_ci_high': or_skin_ci[1],
    },
}

# Print nicely
import json
print(json.dumps(out, indent=2))
