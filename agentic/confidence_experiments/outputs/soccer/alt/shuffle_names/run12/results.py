import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load
df = pd.read_csv('soccer.csv')

# Variables
sub = df.dropna(subset=['rater1', 'nExp', 'redCards', 'yellowCards']).copy()
sub = sub[sub['redCards'] > 0]
sub['skin_tone'] = sub[['rater1', 'nExp']].mean(axis=1)
sub['red_card_count'] = sub['yellowCards'].astype(float)
sub['games'] = sub['redCards'].astype(float)

# Categories for extremes
sub['skin_cat'] = pd.cut(
    sub['skin_tone'],
    bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
    labels=['very_light', 'light', 'medium', 'dark']
)

light = sub[sub['skin_cat'] == 'very_light']
dark = sub[sub['skin_cat'] == 'dark']

light_rate = light['red_card_count'].sum() / light['games'].sum()
dark_rate = dark['red_card_count'].sum() / dark['games'].sum()

# Poisson with continuous skin tone
X = sm.add_constant(sub['skin_tone'])
model = sm.GLM(sub['red_card_count'], X, family=sm.families.Poisson(), offset=np.log(sub['games']))
res = model.fit(cov_type='HC1')
coef = res.params['skin_tone']
se = res.bse['skin_tone']
pval = res.pvalues['skin_tone']
rr = np.exp(coef)
ci = res.conf_int().loc['skin_tone']
rr_ci = np.exp(ci)

# Poisson with dark vs very light
ld = sub[sub['skin_cat'].isin(['very_light','dark'])].copy()
ld['dark_indicator'] = (ld['skin_cat'] == 'dark').astype(int)
X_ld = sm.add_constant(ld['dark_indicator'])
model_ld = sm.GLM(ld['red_card_count'], X_ld, family=sm.families.Poisson(), offset=np.log(ld['games']))
res_ld = model_ld.fit(cov_type='HC1')
coef_ld = res_ld.params['dark_indicator']
se_ld = res_ld.bse['dark_indicator']
pval_ld = res_ld.pvalues['dark_indicator']
rr_ld = np.exp(coef_ld)
rr_ci_ld = np.exp(res_ld.conf_int().loc['dark_indicator'])

print('N total:', len(df))
print('N with skin ratings:', len(sub))
print('Light (very light) dyads:', len(light))
print('Dark dyads:', len(dark))
print('Light rate per game:', light_rate)
print('Dark rate per game:', dark_rate)
print('Rate ratio dark/light:', dark_rate / light_rate)

print('\nPoisson continuous skin tone:')
print('coef', coef, 'se', se, 'p', pval, 'RR', rr, 'RR_CI', tuple(rr_ci))

print('\nPoisson dark vs very light:')
print('coef', coef_ld, 'se', se_ld, 'p', pval_ld, 'RR', rr_ld, 'RR_CI', tuple(rr_ci_ld))
