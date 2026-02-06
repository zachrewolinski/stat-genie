import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Skin tone average (use mean of two raters)
skin_cols = ['feature18', 'feature19']

df['skin_avg'] = df[skin_cols].mean(axis=1)

# Keep rows with skin ratings and valid games
df = df.dropna(subset=['skin_avg', 'feature9', 'feature16'])

# Define light vs dark
# Normalized scale likely: 0,0.25,0.5,0.75,1.0
# Treat <0.5 as light, >0.5 as dark. Exclude exactly 0.5 (neither).

df = df[(df['skin_avg'] < 0.5) | (df['skin_avg'] > 0.5)].copy()

df['dark'] = (df['skin_avg'] > 0.5).astype(int)

# Outcomes and exposure
df['red_cards'] = df['feature16']
df['games'] = df['feature9']

# Aggregate rates
agg = df.groupby('dark').agg(
    red_cards=('red_cards', 'sum'),
    games=('games', 'sum'),
    dyads=('red_cards', 'size')
).reset_index()
agg['rate_per_game'] = agg['red_cards'] / agg['games']

# Poisson regression with log(games) offset
X = sm.add_constant(df['dark'])
model = sm.GLM(df['red_cards'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
res = model.fit()

coef = res.params['dark']
rr = np.exp(coef)
pval = res.pvalues['dark']

# Save results summary for manual inspection
with open('analysis_output.txt', 'w') as f:
    f.write('Aggregated rates by skin tone (0=light,1=dark)\n')
    f.write(agg.to_string(index=False))
    f.write('\n\n')
    f.write('Poisson regression with offset(log(games))\n')
    f.write(res.summary().as_text())
    f.write('\n')
    f.write(f'Rate ratio (dark vs light): {rr:.4f}\n')
    f.write(f'p-value: {pval:.6g}\n')
