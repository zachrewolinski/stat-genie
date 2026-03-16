import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Construct skin tone from the two raters (normalized 0-1)
df['skin'] = df[['rater1', 'nExp']].mean(axis=1)

# Outcome: red cards received in dyad. Based on metadata, 'yellowCards' maps to red cards.
# Exposure: number of games in dyad. 'redCards' appears to be games (min 1, max 47).

# Filter usable rows
use = df[['skin', 'yellowCards', 'redCards']].copy()
use = use[(use['skin'].notna()) & (use['yellowCards'].notna()) & (use['redCards'] > 0)]

# Grouping for light vs dark
use['skin_group'] = np.where(use['skin'] > 0.5, 'dark', 'light')

# Rate per game by group
rate_stats = use.groupby('skin_group').apply(
    lambda g: pd.Series({
        'red_cards': g['yellowCards'].sum(),
        'games': g['redCards'].sum(),
        'rate_per_game': g['yellowCards'].sum() / g['redCards'].sum()
    })
)

# Poisson regression with offset (rate model), continuous skin tone
X = sm.add_constant(use['skin'])
offset = np.log(use['redCards'])
poisson_model = sm.GLM(use['yellowCards'], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

# Binary group model (dark vs light)
use['dark'] = (use['skin_group'] == 'dark').astype(int)
Xg = sm.add_constant(use['dark'])
poisson_g = sm.GLM(use['yellowCards'], Xg, family=sm.families.Poisson(), offset=offset)
poisson_g_res = poisson_g.fit()

# Overdispersion check
overdisp = poisson_res.deviance / poisson_res.df_resid

# Collect results
skin_beta = poisson_res.params['skin']
skin_p = poisson_res.pvalues['skin']
skin_irr = np.exp(skin_beta)
skin_ci = np.exp(poisson_res.conf_int().loc['skin'])

# Group effect
dark_beta = poisson_g_res.params['dark']
dark_p = poisson_g_res.pvalues['dark']
dark_irr = np.exp(dark_beta)
dark_ci = np.exp(poisson_g_res.conf_int().loc['dark'])

print('Rows used:', len(use))
print('\nRate per game by skin group:')
print(rate_stats)
print('\nPoisson (continuous skin):')
print('beta:', skin_beta, 'IRR:', skin_irr, 'CI:', tuple(skin_ci), 'p:', skin_p)
print('overdispersion:', overdisp)

print('\nPoisson (dark vs light):')
print('beta:', dark_beta, 'IRR:', dark_irr, 'CI:', tuple(dark_ci), 'p:', dark_p)
