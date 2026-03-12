import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Identify key variables
# red card counts appear to be in yellowCards (0-2, mean ~0.013)
# games in dyad appear to be redCards (min 1, mean ~2.9)
# skin tone ratings from two raters: rater1 and nExp

# Filter to rows with skin tone ratings and valid games
sub = df.dropna(subset=['rater1', 'nExp', 'redCards', 'yellowCards']).copy()
sub = sub[sub['redCards'] > 0]

# Compute mean skin tone (0=very light, 1=very dark)
sub['skin_tone'] = sub[['rater1', 'nExp']].mean(axis=1)

# Red cards count and exposure
sub['red_card_count'] = sub['yellowCards'].astype(float)
sub['games'] = sub['redCards'].astype(float)

# Basic group comparisons using light vs dark (extremes) and full scale
sub['skin_cat'] = pd.cut(
    sub['skin_tone'],
    bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
    labels=['very_light', 'light', 'medium', 'dark']
)

# Define light vs dark using extremes (very_light vs dark)
light = sub[sub['skin_cat'] == 'very_light']
dark = sub[sub['skin_cat'] == 'dark']

# Compute red card rates per game
light_rate = (light['red_card_count'].sum() / light['games'].sum()) if light['games'].sum() > 0 else np.nan
dark_rate = (dark['red_card_count'].sum() / dark['games'].sum()) if dark['games'].sum() > 0 else np.nan

# Poisson regression with offset log(games)
# Predictor: skin_tone continuous
X = sm.add_constant(sub['skin_tone'])
model = sm.GLM(sub['red_card_count'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
res = model.fit(cov_type='HC1')

# Poisson with light vs dark indicator (extremes)
ld = sub[sub['skin_cat'].isin(['very_light','dark'])].copy()
ld['dark_indicator'] = (ld['skin_cat'] == 'dark').astype(int)
X_ld = sm.add_constant(ld['dark_indicator'])
model_ld = sm.GLM(ld['red_card_count'], X_ld, family=sm.families.Poisson(), offset=np.log(ld['games']))
res_ld = model_ld.fit(cov_type='HC1')

# Summaries
print('Rows total:', len(df))
print('Rows with skin ratings:', len(sub))
print('Skin tone value counts:', sub['skin_tone'].value_counts().sort_index())
print('Skin category counts:', sub['skin_cat'].value_counts())
print('Light rate:', light_rate)
print('Dark rate:', dark_rate)

print('\nPoisson (continuous skin tone):')
print(res.summary().as_text())

print('\nPoisson (dark vs very light):')
print(res_ld.summary().as_text())

# Extract key stats
coef = res.params['skin_tone']
se = res.bse['skin_tone']
pval = res.pvalues['skin_tone']
rate_ratio = np.exp(coef)

coef_ld = res_ld.params['dark_indicator']
se_ld = res_ld.bse['dark_indicator']
pval_ld = res_ld.pvalues['dark_indicator']
rate_ratio_ld = np.exp(coef_ld)

print('\nKey stats:')
print({'coef': coef, 'se': se, 'pval': pval, 'rate_ratio': rate_ratio})
print({'coef_ld': coef_ld, 'se_ld': se_ld, 'pval_ld': pval_ld, 'rate_ratio_ld': rate_ratio_ld})
