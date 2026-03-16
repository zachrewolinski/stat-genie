import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

path = '/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/soccer/alt/shuffle_names/run4/soccer.csv'

df = pd.read_csv(path)

# Identify variables based on distribution (see info.json mapping)
# Skin tone ratings appear in rater1 and nExp (0..1, 5 unique values)
for col in ['rater1', 'nExp', 'yellowCards', 'redCards']:
    if col not in df.columns:
        raise ValueError(f"Missing expected column {col}")

# Mean skin tone (0=very light, 1=very dark)
mean_skin = df[['rater1', 'nExp']].mean(axis=1)

# Red card counts appear in yellowCards (0..2)
red_cards = df['yellowCards']

# Exposure (# games with referee) appears in redCards (1..47)
games = df['redCards']

# Clean
mask = mean_skin.notna() & red_cards.notna() & games.notna() & (games > 0)
clean = df.loc[mask].copy()
clean['mean_skin'] = mean_skin[mask]
clean['red_cards'] = red_cards[mask]
clean['games'] = games[mask]

# Binary dark vs light (threshold at 0.5 = midpoint of 5-point scale)
clean['dark'] = (clean['mean_skin'] >= 0.5).astype(int)

# Group rates
grp = clean.groupby('dark').agg(
    red_cards_sum=('red_cards', 'sum'),
    games_sum=('games', 'sum'),
    n=('red_cards', 'size')
)

grp['rate_per_game'] = grp['red_cards_sum'] / grp['games_sum']

# Poisson regression with exposure
X = sm.add_constant(clean['mean_skin'])
model = sm.GLM(clean['red_cards'], X, family=sm.families.Poisson(), offset=np.log(clean['games']))
res = model.fit()

coef = res.params['mean_skin']
se = res.bse['mean_skin']
pval = res.pvalues['mean_skin']
rate_ratio = float(np.exp(coef))
ci_low, ci_high = np.exp(res.conf_int().loc['mean_skin'].values)

# Also compute group rate ratio (dark vs light)
if 0 in grp.index and 1 in grp.index:
    rr_dark_light = float(grp.loc[1, 'rate_per_game'] / grp.loc[0, 'rate_per_game']) if grp.loc[0, 'rate_per_game'] > 0 else float('inf')
else:
    rr_dark_light = float('nan')

results = {
    'n_rows': int(clean.shape[0]),
    'dark_count': int(clean['dark'].sum()),
    'light_count': int((clean['dark'] == 0).sum()),
    'rate_light': float(grp.loc[0, 'rate_per_game']) if 0 in grp.index else None,
    'rate_dark': float(grp.loc[1, 'rate_per_game']) if 1 in grp.index else None,
    'rate_ratio_dark_light': rr_dark_light,
    'poisson_coef_mean_skin': float(coef),
    'poisson_se_mean_skin': float(se),
    'poisson_pvalue_mean_skin': float(pval),
    'poisson_rate_ratio_per_unit_skin': rate_ratio,
    'poisson_rr_ci_low': float(ci_low),
    'poisson_rr_ci_high': float(ci_high),
}

print(json.dumps(results, indent=2))
