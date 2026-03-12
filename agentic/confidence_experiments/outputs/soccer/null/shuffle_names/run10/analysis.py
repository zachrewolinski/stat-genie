import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Identify key columns based on value ranges in this shuffled dataset
# Skin tone ratings are the two 0-1 scaled 5-level columns
skin1 = df['rater1']
skin2 = df['nExp']
mean_skin = (skin1 + skin2) / 2

# Games per dyad is the integer count with max 47 and no zeros
# In this shuffled dataset that's the column named redCards
# Red card counts appear as small integer counts (max 2 and max 3)
# We'll analyze direct red cards (max 2) and total red (direct + second-yellow)

games = df['redCards']
red_direct = df['yellowCards']
yellow_red = df['meanExp']
red_total = red_direct + yellow_red

# Build analysis dataframe
analysis_df = pd.DataFrame({
    'mean_skin': mean_skin,
    'games': games,
    'red_direct': red_direct,
    'yellow_red': yellow_red,
    'red_total': red_total,
})

# Drop rows with missing or non-positive games
analysis_df = analysis_df.dropna()
analysis_df = analysis_df[analysis_df['games'] > 0]

# Poisson regression with log(games) offset
# red_direct ~ mean_skin
X = sm.add_constant(analysis_df['mean_skin'])

poisson_direct = sm.GLM(
    analysis_df['red_direct'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['games'])
).fit()

poisson_total = sm.GLM(
    analysis_df['red_total'],
    X,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['games'])
).fit()

# Also compute simple rate difference (per game) across skin tone quantiles
analysis_df['skin_quartile'] = pd.qcut(analysis_df['mean_skin'], 4, labels=False, duplicates='drop')
rate_by_quartile = analysis_df.groupby('skin_quartile').apply(
    lambda g: pd.Series({
        'mean_skin': g['mean_skin'].mean(),
        'red_direct_per_game': (g['red_direct'].sum() / g['games'].sum()),
        'red_total_per_game': (g['red_total'].sum() / g['games'].sum()),
        'n': len(g)
    })
).reset_index()

print('Rows used:', len(analysis_df))
print('\nPoisson (direct red) coef for mean_skin:')
print(poisson_direct.params)
print(poisson_direct.bse)
print('p-value:', poisson_direct.pvalues['mean_skin'])
print('IRR:', np.exp(poisson_direct.params['mean_skin']))

print('\nPoisson (total red) coef for mean_skin:')
print(poisson_total.params)
print(poisson_total.bse)
print('p-value:', poisson_total.pvalues['mean_skin'])
print('IRR:', np.exp(poisson_total.params['mean_skin']))

print('\nRate by skin tone quartile:')
print(rate_by_quartile)

