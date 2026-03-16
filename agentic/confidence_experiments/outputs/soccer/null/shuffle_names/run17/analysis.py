import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Skin tone ratings
skin_cols = ['rater1', 'nExp']
df['mean_skin'] = df[skin_cols].mean(axis=1)

# Total red cards: combine two rare red-card-like columns
# Based on distributions, yellowCards (0-2) and meanExp (0-3) appear to be the two red card counts
# (direct red and second yellow). We sum them to represent total red cards.
df['red_total'] = df['yellowCards'] + df['meanExp']

# Games exposure (column with max 47, min 1)
df['games_exposure'] = df['redCards']

# Filter to rows with skin tone and positive games
analysis_df = df.dropna(subset=['mean_skin']).copy()
analysis_df = analysis_df[analysis_df['games_exposure'] > 0]

print('rows with skin tone:', len(analysis_df))
print('mean_skin unique:', sorted(analysis_df['mean_skin'].unique()))

# Summary by skin tone groups
light_mask = analysis_df['mean_skin'] <= 0.25
dark_mask = analysis_df['mean_skin'] >= 0.75

summary = {}
for label, mask in [('light', light_mask), ('dark', dark_mask)]:
    sub = analysis_df[mask]
    summary[label] = {
        'rows': len(sub),
        'players': sub['photoID'].nunique(),  # short name column
        'red_total': sub['red_total'].sum(),
        'games': sub['games_exposure'].sum(),
        'rate_per_game': sub['red_total'].sum() / sub['games_exposure'].sum() if sub['games_exposure'].sum() > 0 else np.nan,
    }
print('group summary', summary)

# Poisson regression: red_total ~ mean_skin with offset log(games)
X = sm.add_constant(analysis_df['mean_skin'])
model = sm.GLM(analysis_df['red_total'], X, family=sm.families.Poisson(), offset=np.log(analysis_df['games_exposure']))
res = model.fit()
print(res.summary())

coef = res.params['mean_skin']
pval = res.pvalues['mean_skin']
rate_ratio = np.exp(coef)
print('coef', coef, 'pval', pval, 'rate_ratio', rate_ratio)

# Predicted rates for light vs dark based on mean_skin 0.25 and 0.75
for val in [0.25, 0.75]:
    pred_rate = np.exp(res.params['const'] + res.params['mean_skin'] * val)
    print('pred rate', val, pred_rate)

# Also test using rate ratio for extreme groups (dark vs light) with Poisson approx
# compute rate ratio and approximate CI
light = analysis_df[light_mask]
dark = analysis_df[dark_mask]

# Only if both groups have events
if light['red_total'].sum() > 0 and dark['red_total'].sum() > 0:
    rate_light = light['red_total'].sum() / light['games_exposure'].sum()
    rate_dark = dark['red_total'].sum() / dark['games_exposure'].sum()
    rr = rate_dark / rate_light
    # standard error for log rate ratio
    se_log_rr = np.sqrt(1/light['red_total'].sum() + 1/dark['red_total'].sum())
    ci_low = np.exp(np.log(rr) - 1.96*se_log_rr)
    ci_high = np.exp(np.log(rr) + 1.96*se_log_rr)
    print('group rate ratio dark/light', rr, 'CI', ci_low, ci_high)
else:
    print('insufficient events for group rate ratio')

