import os
import sys
cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", cwd)]

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Identify variables
# Skin tone ratings (0-1) from two raters
skin1 = df['rater1']
skin2 = df['nExp']

skin_mean = pd.concat([skin1, skin2], axis=1).mean(axis=1, skipna=True)

# Outcomes: red cards (direct + second yellow). These appear to be the rare count columns.
# In this shuffled dataset, meanExp and yellowCards are rare counts (~1% non-zero), consistent with red cards types.
direct_red = df['meanExp']
second_yellow_red = df['yellowCards']
total_red = direct_red + second_yellow_red

# Exposure: number of games in dyad. Column 'redCards' sums wins/losses/ties and ranges 1-47.
games = df['redCards']

# Build analysis frame
analysis = pd.DataFrame({
    'skin_mean': skin_mean,
    'total_red': total_red,
    'direct_red': direct_red,
    'second_yellow_red': second_yellow_red,
    'games': games,
})

# Drop missing skin ratings or nonpositive games
analysis = analysis.dropna(subset=['skin_mean'])
analysis = analysis[analysis['games'] > 0]

# Categorize skin tone
analysis['skin_cat'] = pd.cut(
    analysis['skin_mean'],
    bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
    labels=['light', 'medium_light', 'medium_dark', 'dark']
)

# Light vs dark subset
ld = analysis[analysis['skin_cat'].isin(['light','dark'])].copy()
ld['dark'] = (ld['skin_cat'] == 'dark').astype(int)

# Compute rates per game
rates = ld.groupby('skin_cat').apply(
    lambda g: pd.Series({
        'n': len(g),
        'total_reds': g['total_red'].sum(),
        'games': g['games'].sum(),
        'rate_per_game': g['total_red'].sum() / g['games'].sum()
    })
)

# Poisson regression: total_red ~ dark + offset(log(games))
X = sm.add_constant(ld['dark'])
model = sm.GLM(ld['total_red'], X, family=sm.families.Poisson(), offset=np.log(ld['games']))
res = model.fit(cov_type='HC0')

# Continuous skin tone regression
analysis['skin_mean_centered'] = analysis['skin_mean'] - analysis['skin_mean'].mean()
Xc = sm.add_constant(analysis['skin_mean_centered'])
model_c = sm.GLM(analysis['total_red'], Xc, family=sm.families.Poisson(), offset=np.log(analysis['games']))
res_c = model_c.fit(cov_type='HC0')

# Summarize
print('Light vs dark rates:')
print(rates)

print('\nPoisson (dark vs light):')
print(res.params)
print(res.bse)
print(res.pvalues)

irr = np.exp(res.params['dark'])
print('IRR dark vs light:', irr)

print('\nPoisson (continuous skin):')
print(res_c.params)
print(res_c.bse)
print(res_c.pvalues)

irr_c = np.exp(res_c.params['skin_mean_centered'])
print('IRR per 1.0 skin unit:', irr_c)

