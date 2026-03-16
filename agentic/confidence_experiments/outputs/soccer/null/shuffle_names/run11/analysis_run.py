import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('soccer.csv')

# Identify columns based on distribution
# Skin tone ratings are rater1 and nExp (both 0-1 in steps of 0.25)
skin = df[['rater1', 'nExp']].mean(axis=1)

# Games in dyad appears to be 'redCards' (min 1, max 47, mean ~2.9)
games = df['redCards']

# Red cards appear to be split across two rare-count columns (meanExp and yellowCards)
# Sum them to count total red cards (straight + second yellow)
red_total = df['meanExp'] + df['yellowCards']

analysis_df = pd.DataFrame({
    'skin': skin,
    'games': games,
    'red_total': red_total,
})

# Clean
analysis_df = analysis_df.dropna()
analysis_df = analysis_df[analysis_df['games'] > 0]

# Basic checks
print('rows', len(analysis_df))
print('red_total max', analysis_df['red_total'].max())
print('games max', analysis_df['games'].max())
print('prop red_total <= games', (analysis_df['red_total'] <= analysis_df['games']).mean())

# Overall rate
analysis_df['rate'] = analysis_df['red_total'] / analysis_df['games']
print('overall red card rate per game', analysis_df['rate'].mean())

# Group comparison: light vs dark
light = analysis_df[analysis_df['skin'] <= 0.25]
dark = analysis_df[analysis_df['skin'] >= 0.75]

light_rate = light['red_total'].sum() / light['games'].sum()
dark_rate = dark['red_total'].sum() / dark['games'].sum()
rate_ratio = dark_rate / light_rate if light_rate > 0 else np.nan

print('light n', len(light), 'dark n', len(dark))
print('light rate', light_rate)
print('dark rate', dark_rate)
print('rate ratio dark/light', rate_ratio)

# Poisson regression with offset for games
X = sm.add_constant(analysis_df['skin'])
model = sm.GLM(analysis_df['red_total'], X, family=sm.families.Poisson(), offset=np.log(analysis_df['games']))
res = model.fit()

beta = res.params['skin']
se = res.bse['skin']
pval = res.pvalues['skin']
irr = np.exp(beta)
ci_low, ci_high = np.exp(res.conf_int().loc['skin'])

print('\nPoisson regression: red_total ~ skin + offset(log(games))')
print('beta', beta, 'se', se, 'pval', pval)
print('IRR', irr, 'CI', (ci_low, ci_high))

# Negative binomial for robustness
nb_model = sm.GLM(analysis_df['red_total'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(analysis_df['games']))
nb_res = nb_model.fit()
nb_beta = nb_res.params['skin']
nb_pval = nb_res.pvalues['skin']
nb_irr = np.exp(nb_beta)
nb_ci_low, nb_ci_high = np.exp(nb_res.conf_int().loc['skin'])
print('\nNegBin regression:')
print('beta', nb_beta, 'pval', nb_pval)
print('IRR', nb_irr, 'CI', (nb_ci_low, nb_ci_high))

# Also check single columns separately (meanExp and yellowCards)
for col in ['meanExp', 'yellowCards']:
    y = df[col]
    tmp = pd.DataFrame({'y': y, 'skin': skin, 'games': games}).dropna()
    tmp = tmp[tmp['games'] > 0]
    X2 = sm.add_constant(tmp['skin'])
    m = sm.GLM(tmp['y'], X2, family=sm.families.Poisson(), offset=np.log(tmp['games']))
    r = m.fit()
    irr2 = np.exp(r.params['skin'])
    ci2 = np.exp(r.conf_int().loc['skin'])
    print(f"\n{col} poisson IRR", irr2, "CI", tuple(ci2), "p", r.pvalues['skin'])
