import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'soccer.csv'

df = pd.read_csv(path)

# skin tone as mean of the two 0-1 rating columns
# Columns with 0-1 ratings: rater1 and nExp
skin = df[['rater1','nExp']].mean(axis=1)

# define dark vs light
# Exclude neutral (==0.5) for a clean contrast
light_mask = skin < 0.5
# dark if > 0.5
dark_mask = skin > 0.5

# games exposure: column with 1-47 (redCards)
games = df['redCards'].astype(float)

# candidate red card counts
red_candidates = {
    'yellowCards': df['yellowCards'],
    'meanExp': df['meanExp'],
    'sum_yellowCards_meanExp': df['yellowCards'] + df['meanExp'],
}

results = {}

for name, red in red_candidates.items():
    # filter rows with exposure >=1 and in light or dark groups
    mask = (games >= 1) & (light_mask | dark_mask)
    sub = df.loc[mask].copy()
    sub['skin'] = skin[mask]
    sub['dark'] = (sub['skin'] > 0.5).astype(int)
    sub['red'] = red[mask].astype(float)
    sub['games'] = games[mask]

    # group rates
    rates = {}
    for grp, gmask in [('light', sub['dark']==0), ('dark', sub['dark']==1)]:
        total_red = sub.loc[gmask, 'red'].sum()
        total_games = sub.loc[gmask, 'games'].sum()
        rate = total_red / total_games if total_games > 0 else np.nan
        rates[grp] = {'total_red': float(total_red), 'total_games': float(total_games), 'rate': float(rate)}

    # Poisson regression with offset log(games)
    X = sm.add_constant(sub['dark'])
    model = sm.GLM(sub['red'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
    try:
        res = model.fit()
        coef = res.params['dark']
        pval = res.pvalues['dark']
        irr = float(np.exp(coef))
    except Exception as e:
        coef = np.nan
        pval = np.nan
        irr = np.nan

    results[name] = {
        'n': int(mask.sum()),
        'rates': rates,
        'coef': float(coef),
        'irr': float(irr),
        'pval': float(pval),
    }

# Also do continuous skin tone model for main candidate (yellowCards)
mask = (games >= 1) & skin.notna()
sub = df.loc[mask].copy()
sub['skin'] = skin[mask]
sub['red'] = df['yellowCards'][mask].astype(float)
sub['games'] = games[mask]
X = sm.add_constant(sub['skin'])
model = sm.GLM(sub['red'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
res = model.fit()
continuous = {
    'coef_skin': float(res.params['skin']),
    'irr_skin': float(np.exp(res.params['skin'])),
    'pval_skin': float(res.pvalues['skin']),
}

print('RESULTS')
for k,v in results.items():
    print(k, v)
print('continuous', continuous)
