import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
Df = pd.read_csv('soccer.csv')

# Skin tone: average of two raters (0=very light ... 1=very dark)
skin_mean = Df[['rater1', 'rater2']].mean(axis=1)

# Define dark vs light using the midpoint (>= 0.5 is darker than neutral)
Df['is_dark'] = (skin_mean >= 0.5).astype(int)

# Aggregate red card rates per game by group
summary = (
    Df.groupby('is_dark')
      .agg(total_red=('redCards', 'sum'), total_games=('games', 'sum'))
      .assign(red_per_game=lambda x: x['total_red'] / x['total_games'])
)

# Poisson regression with exposure (games) as offset
X = sm.add_constant(Df['is_dark'])
model = sm.GLM(Df['redCards'], X, family=sm.families.Poisson(), offset=np.log(Df['games']))
res = model.fit(cov_type='HC0')

irr = float(np.exp(res.params['is_dark']))
p_value = float(res.pvalues['is_dark'])

print('Group rates (red cards per game):')
print(summary)
print('\nPoisson regression (offset=log(games))')
print(res.summary().tables[1])
print(f"\nIncidence rate ratio (dark vs light): {irr:.4f}")
print(f"p-value: {p_value:.4g}")

# Save key results for conclusion
summary.to_csv('group_rates.csv')
