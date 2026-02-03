import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

# Compute average skin tone from two raters
skin = df[['rater1', 'rater2']].mean(axis=1)
df = df.assign(skin=skin)

# Basic cleaning: require skin tone, games > 0, redCards not null
clean = df.dropna(subset=['skin', 'games', 'redCards']).copy()
clean = clean[clean['games'] > 0]

# Create dark/light groups using the 5-point scale (0, .25, .5, .75, 1)
# Define light as <= 0.25 and dark as >= 0.75
clean['skin_group'] = np.select(
    [clean['skin'] <= 0.25, clean['skin'] >= 0.75],
    ['light', 'dark'],
    default='mid'
)

# Compute red card rates per game by group
summary = (
    clean.groupby('skin_group')
    .apply(lambda g: pd.Series({
        'dyads': len(g),
        'total_games': g['games'].sum(),
        'total_reds': g['redCards'].sum(),
        'reds_per_game': g['redCards'].sum() / g['games'].sum()
    }))
)

# Poisson regression: redCards ~ skin with log(games) offset
# This models red cards per game as a function of continuous skin tone
X = sm.add_constant(clean['skin'])
y = clean['redCards']
offset = np.log(clean['games'])
poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_results = poisson_model.fit()

# Save key results for inspection
coef = poisson_results.params['skin']
pval = poisson_results.pvalues['skin']

print('Group summary (light/dark/mid):')
print(summary)
print('\nPoisson regression (redCards ~ skin with log(games) offset):')
print(poisson_results.summary().tables[1])
print(f"\nSkin coefficient: {coef:.4f}, p-value: {pval:.4g}")

# Save a compact results file for reference
with open('analysis_results.txt', 'w') as f:
    f.write('Group summary (light/dark/mid):\n')
    f.write(summary.to_string())
    f.write('\n\nPoisson regression (redCards ~ skin with log(games) offset):\n')
    f.write(poisson_results.summary().tables[1].as_text())
    f.write(f"\n\nSkin coefficient: {coef:.4f}, p-value: {pval:.4g}\n")
