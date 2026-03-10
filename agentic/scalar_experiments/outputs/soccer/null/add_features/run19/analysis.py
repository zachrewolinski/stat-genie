import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Ensure numeric columns for analysis
for col in ['rater1', 'rater2', 'redCards', 'games']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute skin tone as mean of available raters
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
# Keep rows with at least one rating and valid games
mask = skin.notna() & df['games'].notna() & (df['games'] > 0) & df['redCards'].notna()
sub = df.loc[mask].copy()
sub['skin_tone'] = skin[mask]

# Define light vs dark (exclude neutral 0.5)
sub['tone_group'] = np.where(sub['skin_tone'] > 0.5, 'dark',
                             np.where(sub['skin_tone'] < 0.5, 'light', 'neutral'))
sub_binary = sub[sub['tone_group'] != 'neutral'].copy()

# Basic counts and rates
summary = {}
summary['n_rows_total'] = int(len(df))
summary['n_rows_skin'] = int(len(sub))
summary['n_rows_binary'] = int(len(sub_binary))

rates = {}
for grp, gdf in sub_binary.groupby('tone_group'):
    total_red = gdf['redCards'].sum()
    total_games = gdf['games'].sum()
    rate = total_red / total_games if total_games > 0 else np.nan
    rates[grp] = {
        'rows': int(len(gdf)),
        'players': int(gdf['playerShort'].nunique()),
        'total_red': float(total_red),
        'total_games': float(total_games),
        'rate_per_game': float(rate),
        'rate_per_10_games': float(rate * 10.0)
    }
summary['rates'] = rates

# Poisson regression: redCards ~ dark + offset(log(games))
# Use dark indicator (1=dark, 0=light)
sub_binary['dark'] = (sub_binary['tone_group'] == 'dark').astype(int)
X = sm.add_constant(sub_binary['dark'])
# offset for exposure
offset = np.log(sub_binary['games'])
model = sm.GLM(sub_binary['redCards'], X, family=sm.families.Poisson(), offset=offset)
res = model.fit(cov_type='HC0')

coef = res.params['dark']
se = res.bse['dark']
pval = res.pvalues['dark']
rate_ratio = float(np.exp(coef))
# 95% CI
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))
summary['poisson_dark_vs_light'] = {
    'coef': float(coef),
    'se': float(se),
    'p_value': float(pval),
    'rate_ratio': rate_ratio,
    'ci_low': ci_low,
    'ci_high': ci_high
}

# Poisson regression with continuous skin_tone
X2 = sm.add_constant(sub['skin_tone'])
model2 = sm.GLM(sub['redCards'], X2, family=sm.families.Poisson(), offset=np.log(sub['games']))
res2 = model2.fit(cov_type='HC0')
coef2 = res2.params['skin_tone']
se2 = res2.bse['skin_tone']
pval2 = res2.pvalues['skin_tone']
rr2 = float(np.exp(coef2))
ci2_low = float(np.exp(coef2 - 1.96 * se2))
ci2_high = float(np.exp(coef2 + 1.96 * se2))
summary['poisson_continuous'] = {
    'coef': float(coef2),
    'se': float(se2),
    'p_value': float(pval2),
    'rate_ratio_per_unit': rr2,
    'ci_low': ci2_low,
    'ci_high': ci2_high
}

# Save summary for inspection
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
