import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Keep rows with skin ratings and games
skin_cols = ['rater1', 'rater2']
for col in skin_cols + ['games', 'redCards']:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df = df.copy()

df['skin_mean'] = df[skin_cols].mean(axis=1)

# Drop rows without skin ratings or games
mask = df['skin_mean'].notna() & df['games'].notna() & df['redCards'].notna()
df = df.loc[mask].copy()

# Remove any non-positive games to avoid log issues
# (games should be >=1 per metadata)
df = df.loc[df['games'] > 0].copy()

# Define skin tone groups: dark if mean >= 0.5, else light
# 0-1 scale; 0=very light, 1=very dark
threshold = 0.5
df['dark_skin'] = (df['skin_mean'] >= threshold).astype(int)

# Aggregate rates
agg = df.groupby('dark_skin').agg(
    dyads=('redCards', 'size'),
    total_red=('redCards', 'sum'),
    total_games=('games', 'sum')
).reset_index()
agg['red_per_game'] = agg['total_red'] / agg['total_games']

# Poisson regression with offset for games
# redCards ~ dark_skin, offset log(games)
X = sm.add_constant(df['dark_skin'])
model = sm.GLM(df['redCards'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
result = model.fit()

# Extract rate ratio
coef = result.params['dark_skin']
rr = float(np.exp(coef))
ci = result.conf_int().loc['dark_skin']
rr_ci = tuple(np.exp(ci))

# Save key outputs
print('Rows used:', len(df))
print('Threshold:', threshold)
print(agg)
print('Rate ratio (dark vs light):', rr)
print('95% CI:', rr_ci)
print('p-value:', result.pvalues['dark_skin'])
