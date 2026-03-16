import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Ensure numeric
for col in ['redCards','games','rater1','rater2']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Mean skin tone across raters (use available rater if one missing)
_df['meanSkin'] = _df[['rater1','rater2']].mean(axis=1, skipna=True)

# Filter to rows with skin tone, games, redCards
df = _df.dropna(subset=['meanSkin','games','redCards']).copy()

# Basic counts
print('Rows with skin tone:', len(df))
print('MeanSkin unique values (sorted):', sorted(df['meanSkin'].dropna().unique())[:10], '... total', df['meanSkin'].nunique())

# Define light/dark thresholds
light_thr = 0.25
dark_thr = 0.75

# Clean in case of float rounding

def classify_skin(x):
    if x <= light_thr:
        return 'light'
    if x >= dark_thr:
        return 'dark'
    return 'mid'


df['skin_cat'] = df['meanSkin'].apply(classify_skin)

print('Skin cat counts:\n', df['skin_cat'].value_counts(dropna=False))

# Aggregate rates by category
agg = df.groupby('skin_cat').agg(redCards=('redCards','sum'), games=('games','sum'))
agg['rate_per_game'] = agg['redCards'] / agg['games']
print('\nAggregate rates by skin_cat:\n', agg)

# Poisson regression with offset for games, dark vs light only
subset = df[df['skin_cat'].isin(['light','dark'])].copy()
subset['dark'] = (subset['skin_cat'] == 'dark').astype(int)
subset = subset[subset['games'] > 0]

# Poisson GLM
X = sm.add_constant(subset['dark'])
model = sm.GLM(subset['redCards'], X, family=sm.families.Poisson(), offset=np.log(subset['games']))
res = model.fit()
print('\nPoisson GLM (dark vs light)')
print(res.summary())

# Rate ratio and CI
coef = res.params['dark']
se = res.bse['dark']
rr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print('\nRate ratio dark vs light:', rr)
print('95% CI:', (ci_low, ci_high))

# Also test continuous meanSkin
X2 = sm.add_constant(df['meanSkin'])
model2 = sm.GLM(df['redCards'], X2, family=sm.families.Poisson(), offset=np.log(df['games']))
res2 = model2.fit()
print('\nPoisson GLM (continuous meanSkin)')
print(res2.summary())

coef2 = res2.params['meanSkin']
se2 = res2.bse['meanSkin']
rr2 = np.exp(coef2)
ci2 = (np.exp(coef2-1.96*se2), np.exp(coef2+1.96*se2))
print('\nRate ratio per +1 skin tone:', rr2)
print('95% CI:', ci2)

# Simple rate ratio by aggregating
light = agg.loc['light'] if 'light' in agg.index else None
dark = agg.loc['dark'] if 'dark' in agg.index else None
if light is not None and dark is not None:
    rr_simple = (dark['redCards']/dark['games']) / (light['redCards']/light['games'])
    print('\nSimple rate ratio (aggregate):', rr_simple)
