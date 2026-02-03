import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Map columns based on provided metadata and observed distributions
# rater1 and nExp are 0-1 skin tone ratings; yellowCards is red card count; redCards is games count
skin1 = _df['rater1']
skin2 = _df['nExp']
red_cards = _df['yellowCards']
games = _df['redCards']

# Build mean skin tone when both ratings available
mean_skin = (skin1 + skin2) / 2.0

# Define light vs dark using ends of the 5-point scale
# 0.0-0.25: light/very light; 0.75-1.0: dark/very dark
light_mask = mean_skin <= 0.25
dark_mask = mean_skin >= 0.75

# Keep only rows with defined light or dark and valid counts
mask = (light_mask | dark_mask) & red_cards.notna() & games.notna()

df = pd.DataFrame({
    'dark': dark_mask[mask].astype(int),
    'red_cards': red_cards[mask].astype(float),
    'games': games[mask].astype(float)
}).copy()

# Aggregate rates
summary = df.groupby('dark').agg(
    total_red_cards=('red_cards', 'sum'),
    total_games=('games', 'sum'),
    n=('red_cards', 'size')
)
summary['rate_per_game'] = summary['total_red_cards'] / summary['total_games']

# Poisson regression with exposure (games) to estimate rate ratio
X = sm.add_constant(df['dark'])
model = sm.GLM(df['red_cards'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
res = model.fit()

beta = res.params['dark']
se = res.bse['dark']
rate_ratio = float(np.exp(beta))
ci_low = float(np.exp(beta - 1.96 * se))
ci_high = float(np.exp(beta + 1.96 * se))
p_value = float(res.pvalues['dark'])

# Save outputs for downstream use
summary.to_csv('analysis_summary.csv')

with open('analysis_results.txt', 'w') as f:
    f.write(f"rate_light={summary.loc[0, 'rate_per_game'] if 0 in summary.index else np.nan}\n")
    f.write(f"rate_dark={summary.loc[1, 'rate_per_game'] if 1 in summary.index else np.nan}\n")
    f.write(f"rate_ratio={rate_ratio}\n")
    f.write(f"ci_low={ci_low}\n")
    f.write(f"ci_high={ci_high}\n")
    f.write(f"p_value={p_value}\n")
