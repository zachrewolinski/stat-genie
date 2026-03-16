import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute average skin tone if both raters available
# rater1 and rater2 are already normalized to 0-1.

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Keep relevant columns
cols = ['skin_tone', 'rater1', 'rater2', 'redCards', 'games', 'yellowCards', 'yellowReds', 'playerShort']
for c in cols:
    if c not in df.columns:
        print('Missing column', c)

# Filter rows with non-missing skin_tone and games and redCards
sub = df[['skin_tone', 'redCards', 'games']].copy()
sub = sub.dropna(subset=['skin_tone', 'redCards', 'games'])
sub = sub[sub['games'] > 0]

# Compute red card rate per game
sub['red_rate'] = sub['redCards'] / sub['games']

# Define light vs dark groups using quartiles (25th and 75th percentiles)
q25 = sub['skin_tone'].quantile(0.25)
q75 = sub['skin_tone'].quantile(0.75)

light = sub[sub['skin_tone'] <= q25].copy()
dark = sub[sub['skin_tone'] >= q75].copy()

# Basic group stats
stats = {
    'n_total': int(len(sub)),
    'n_light': int(len(light)),
    'n_dark': int(len(dark)),
    'q25': float(q25),
    'q75': float(q75),
    'mean_red_rate_light': float(light['red_rate'].mean()),
    'mean_red_rate_dark': float(dark['red_rate'].mean()),
    'mean_red_rate_all': float(sub['red_rate'].mean()),
    'mean_skin_tone': float(sub['skin_tone'].mean()),
}

# Poisson regression with offset for games (all data): redCards ~ skin_tone
# Use robust (HC1) standard errors for mild overdispersion
model = smf.glm('redCards ~ skin_tone', data=sub, family=sm.families.Poisson(), offset=np.log(sub['games']))
res = model.fit(cov_type='HC1')

coef = res.params['skin_tone']
se = res.bse['skin_tone']
rr = float(np.exp(coef))
# 95% CI for rate ratio
ci_low, ci_high = np.exp(res.conf_int().loc['skin_tone'])

stats.update({
    'coef_skin_tone': float(coef),
    'se_skin_tone': float(se),
    'p_value_skin_tone': float(res.pvalues['skin_tone']),
    'rr_skin_tone': float(rr),
    'rr_ci_low': float(ci_low),
    'rr_ci_high': float(ci_high),
})

# Two-group Poisson model (dark vs light) for rate ratio
sub2 = sub.copy()
sub2['dark_group'] = (sub2['skin_tone'] >= q75).astype(int)
sub2['light_group'] = (sub2['skin_tone'] <= q25).astype(int)
sub2 = sub2[(sub2['dark_group'] == 1) | (sub2['light_group'] == 1)].copy()
sub2['dark'] = sub2['dark_group']

model2 = smf.glm('redCards ~ dark', data=sub2, family=sm.families.Poisson(), offset=np.log(sub2['games']))
res2 = model2.fit(cov_type='HC1')

coef2 = res2.params['dark']
rr2 = float(np.exp(coef2))
ci2_low, ci2_high = np.exp(res2.conf_int().loc['dark'])

stats.update({
    'rr_dark_vs_light': float(rr2),
    'rr_dark_vs_light_ci_low': float(ci2_low),
    'rr_dark_vs_light_ci_high': float(ci2_high),
    'p_value_dark_vs_light': float(res2.pvalues['dark']),
})

# Logistic regression: any red card in dyad
sub['any_red'] = (sub['redCards'] > 0).astype(int)
logit_model = smf.glm('any_red ~ skin_tone', data=sub, family=sm.families.Binomial(), freq_weights=sub['games'])
logit_res = logit_model.fit(cov_type='HC1')

coef3 = logit_res.params['skin_tone']
or3 = float(np.exp(coef3))
ci3_low, ci3_high = np.exp(logit_res.conf_int().loc['skin_tone'])

stats.update({
    'or_skin_tone_any_red': float(or3),
    'or_ci_low': float(ci3_low),
    'or_ci_high': float(ci3_high),
    'p_value_any_red': float(logit_res.pvalues['skin_tone']),
})

print(json.dumps(stats, indent=2))
