import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

# Compute mean skin tone per row
# Some rows might have missing in feature18/19; compute mean of available
skin = df[['feature18', 'feature19']].mean(axis=1, skipna=True)
df = df.assign(skin_tone=skin)

# Basic missingness
print('rows:', len(df))
print('skin_tone missing:', df['skin_tone'].isna().mean())
print('red_cards missing:', df['feature16'].isna().mean())
print('games missing:', df['feature9'].isna().mean())

# Aggregate to player level (feature1 short name)
player_cols = ['feature1']
agg = (
    df.dropna(subset=['skin_tone'])
      .groupby('feature1', as_index=False)
      .agg(
          skin_tone=('skin_tone', 'mean'),
          red_cards=('feature16', 'sum'),
          games=('feature9', 'sum'),
      )
)

# Filter out zero games or missing
agg = agg[(agg['games'] > 0) & agg['red_cards'].notna() & agg['skin_tone'].notna()]

print('players with skin tone:', len(agg))

# Create rate
agg['red_per_game'] = agg['red_cards'] / agg['games']

# Define light/dark groups using 5-point scale thresholds
light = agg[agg['skin_tone'] <= 0.25]
dark = agg[agg['skin_tone'] >= 0.75]
mid = agg[(agg['skin_tone'] > 0.25) & (agg['skin_tone'] < 0.75)]

print('light players:', len(light), 'dark players:', len(dark), 'mid players:', len(mid))

# Compare mean rates and totals
for name, grp in [('light', light), ('dark', dark), ('mid', mid), ('all', agg)]:
    print(name, 'mean red_per_game', grp['red_per_game'].mean(),
          'median', grp['red_per_game'].median(),
          'mean red_cards', grp['red_cards'].mean(),
          'mean games', grp['games'].mean())

# Poisson regression at player level with offset log(games)
# Use skin_tone continuous
X = sm.add_constant(agg['skin_tone'])
y = agg['red_cards']
offset = np.log(agg['games'])
model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
res = model.fit(cov_type='HC0')
print('\nPoisson GLM (player-level)')
print(res.summary())

# Compute incidence rate ratio for skin_tone (0 to 1)
coef = res.params['skin_tone']
se = res.bse['skin_tone']
irr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print('IRR per +1 skin_tone:', irr, '95% CI', ci_low, ci_high)

# Also test light vs dark via Poisson with binary indicator
agg['dark'] = (agg['skin_tone'] >= 0.75).astype(int)
agg['light'] = (agg['skin_tone'] <= 0.25).astype(int)
ld = agg[(agg['dark'] == 1) | (agg['light'] == 1)].copy()
ld['dark_indicator'] = ld['dark']
X2 = sm.add_constant(ld['dark_indicator'])
y2 = ld['red_cards']
offset2 = np.log(ld['games'])
model2 = sm.GLM(y2, X2, family=sm.families.Poisson(), offset=offset2)
res2 = model2.fit(cov_type='HC0')
print('\nPoisson GLM light vs dark')
print(res2.summary())
coef2 = res2.params['dark_indicator']
se2 = res2.bse['dark_indicator']
irr2 = np.exp(coef2)
ci_low2 = np.exp(coef2 - 1.96*se2)
ci_high2 = np.exp(coef2 + 1.96*se2)
print('IRR dark vs light:', irr2, '95% CI', ci_low2, ci_high2)

# Non-parametric test of rates (Mann-Whitney)
try:
    from scipy.stats import mannwhitneyu
    if len(light) > 0 and len(dark) > 0:
        stat, p = mannwhitneyu(dark['red_per_game'], light['red_per_game'], alternative='two-sided')
        print('Mann-Whitney U p-value (dark vs light rates):', p)
except Exception as e:
    print('Mann-Whitney failed:', e)

# Also evaluate dyad-level model as sensitivity
# Poisson at dyad level with offset log(games)
dyad = df.dropna(subset=['skin_tone', 'feature16', 'feature9']).copy()
X3 = sm.add_constant(dyad['skin_tone'])
y3 = dyad['feature16']
offset3 = np.log(dyad['feature9'])
model3 = sm.GLM(y3, X3, family=sm.families.Poisson(), offset=offset3)
res3 = model3.fit(cov_type='HC0')
print('\nPoisson GLM (dyad-level)')
print(res3.summary())
coef3 = res3.params['skin_tone']
se3 = res3.bse['skin_tone']
irr3 = np.exp(coef3)
ci_low3 = np.exp(coef3 - 1.96*se3)
ci_high3 = np.exp(coef3 + 1.96*se3)
print('Dyad IRR per +1 skin_tone:', irr3, '95% CI', ci_low3, ci_high3)
