import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Column mapping inferred from distributions
red_card_col = 'yellowCards'  # actual red cards (0-2)
games_col = 'redCards'        # actual games in dyad (>=1)

# Skin tone from two raters (0-1 scale)
skin_cols = ['rater1', 'nExp']

# Clean dataset
sub = df[[red_card_col, games_col] + skin_cols].copy()
sub = sub.dropna()

# Average skin tone
sub['skin_tone'] = sub[skin_cols].mean(axis=1)

# Ensure numeric and positive exposure
sub = sub[(sub[games_col] > 0) & (sub[red_card_col] >= 0)]

# Rate per game for summaries
sub['red_rate'] = sub[red_card_col] / sub[games_col]

# Summary by skin tone category (5-point scale mapped to 0-1)
# Values are multiples of 0.25; use rounding to nearest 0.25 for grouping
sub['skin_cat'] = (sub['skin_tone'] * 4).round() / 4

summary = sub.groupby('skin_cat').agg(
    n=('skin_tone', 'size'),
    red_cards=(red_card_col, 'sum'),
    games=(games_col, 'sum'),
)
summary['rate_per_game'] = summary['red_cards'] / summary['games']

# Light vs dark comparison
light = sub[sub['skin_tone'] <= 0.25]
dark = sub[sub['skin_tone'] >= 0.75]

def rate(df_part):
    return df_part[red_card_col].sum() / df_part[games_col].sum()

light_rate = rate(light)
dark_rate = rate(dark)

# Poisson regression with offset for games
# Model: red_cards ~ skin_tone + offset(log(games))
X = sm.add_constant(sub['skin_tone'])
model = sm.GLM(sub[red_card_col], X, family=sm.families.Poisson(), offset=np.log(sub[games_col]))
res = model.fit(cov_type='HC0')

coef = res.params['skin_tone']
se = res.bse['skin_tone']
wald_z = coef / se
p_value = res.pvalues['skin_tone']
rate_ratio = float(np.exp(coef))

# Save key results
results = {
    'n_rows': int(sub.shape[0]),
    'summary_by_skin_cat': summary.reset_index().to_dict(orient='list'),
    'light_rate': float(light_rate),
    'dark_rate': float(dark_rate),
    'poisson_coef': float(coef),
    'poisson_se': float(se),
    'poisson_p_value': float(p_value),
    'poisson_rate_ratio': float(rate_ratio),
}

# Write results for inspection
pd.DataFrame([{
    'n_rows': results['n_rows'],
    'light_rate': results['light_rate'],
    'dark_rate': results['dark_rate'],
    'poisson_coef': results['poisson_coef'],
    'poisson_se': results['poisson_se'],
    'poisson_p_value': results['poisson_p_value'],
    'poisson_rate_ratio': results['poisson_rate_ratio'],
}]).to_csv('analysis_results.csv', index=False)

# Also dump summary to stdout
print('Rows used:', results['n_rows'])
print('Light rate:', light_rate)
print('Dark rate:', dark_rate)
print('Poisson coef:', coef)
print('Rate ratio per +1 skin_tone:', rate_ratio)
print('p-value:', p_value)
print('\nSummary by skin category:')
print(summary)
