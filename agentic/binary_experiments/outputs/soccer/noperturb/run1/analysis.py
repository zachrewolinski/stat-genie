import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('soccer.csv')

# Compute mean skin tone (0=very light, 1=very dark)
_df['skin_mean'] = _df[['rater1', 'rater2']].mean(axis=1)

# Aggregate to player level to avoid dyad duplication
player = (
    _df.dropna(subset=['skin_mean'])
        .groupby('playerShort', as_index=False)
        .agg(
            skin_mean=('skin_mean', 'mean'),
            redCards=('redCards', 'sum'),
            games=('games', 'sum')
        )
)

# Avoid division by zero
player = player[player['games'] > 0].copy()
player['red_rate'] = player['redCards'] / player['games']

# Define light vs dark using midpoint threshold 0.5
player['skin_group'] = np.where(player['skin_mean'] > 0.5, 'dark', 'light_or_mid')

summary = player.groupby('skin_group').agg(
    players=('playerShort', 'count'),
    total_reds=('redCards', 'sum'),
    total_games=('games', 'sum'),
    mean_red_rate=('red_rate', 'mean')
).reset_index()
summary['reds_per_game'] = summary['total_reds'] / summary['total_games']

print('Player-level summary by skin group (threshold 0.5):')
print(summary.to_string(index=False))

# Poisson regression: redCards ~ skin_mean with log(games) offset
# This models red card counts per exposure (games)
player['log_games'] = np.log(player['games'])

model = smf.glm('redCards ~ skin_mean', data=player, family=sm.families.Poisson(), offset=player['log_games'])
result = model.fit()

print('\nPoisson regression (redCards ~ skin_mean, offset log(games))')
print(result.summary())

coef = result.params['skin_mean']
pval = result.pvalues['skin_mean']

# Exponentiated effect for interpretability
rate_ratio = np.exp(coef)

print(f"\nSkin tone coefficient: {coef:.4f}, rate ratio: {rate_ratio:.3f}, p-value: {pval:.4g}")

# Save key outputs for conclusion
with open('analysis_results.txt', 'w') as f:
    f.write('Player-level summary by skin group (threshold 0.5):\n')
    f.write(summary.to_string(index=False))
    f.write('\n\nPoisson regression (redCards ~ skin_mean, offset log(games))\n')
    f.write(result.summary().as_text())
    f.write(f"\n\nSkin tone coefficient: {coef:.4f}, rate ratio: {rate_ratio:.3f}, p-value: {pval:.4g}\n")
