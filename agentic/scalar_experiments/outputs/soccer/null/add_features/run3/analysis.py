import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'

df = pd.read_csv(path)

# compute skin tone average
# rater1 and rater2 are on 0-1 scale
# use mean of available raters
skin = df[['rater1','rater2']].mean(axis=1, skipna=True)

# assign
# if both missing -> NaN
# define dark > 0.5, light <= 0.5
# also record numeric skin

df = df.copy()

df['skin_avg'] = skin

# count missing skin
missing_skin = df['skin_avg'].isna().sum()

# binary

df['skin_dark'] = np.where(df['skin_avg']>0.5, 1, np.where(df['skin_avg'].notna(),0,np.nan))

# summary counts

summary = df[['skin_avg','skin_dark','redCards','games']].copy()

# Filter rows with skin
f = df['skin_dark'].notna() & df['games'].notna() & df['redCards'].notna()

sub = df.loc[f].copy()

# compute red cards per game by group

agg = sub.groupby('skin_dark').apply(lambda g: pd.Series({
    'n_dyads': len(g),
    'total_games': g['games'].sum(),
    'total_red': g['redCards'].sum(),
    'red_per_game': g['redCards'].sum()/g['games'].sum() if g['games'].sum()>0 else np.nan,
    'mean_red_per_dyad': g['redCards'].mean(),
})).rename(index={0:'light',1:'dark'})

# Poisson regression with offset log(games)
# For dyads with games>0

sub2 = sub[sub['games']>0].copy()

# Add intercept and predictor
# Using statsmodels GLM

model = smf.glm('redCards ~ skin_dark', data=sub2, family=sm.families.Poisson(), offset=np.log(sub2['games']))
res = model.fit()

# Rate ratio and CI
coef = res.params['skin_dark']
se = res.bse['skin_dark']
rr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

# p-value
pval = res.pvalues['skin_dark']

# also try using continuous skin_avg
model_cont = smf.glm('redCards ~ skin_avg', data=sub2, family=sm.families.Poisson(), offset=np.log(sub2['games']))
res_cont = model_cont.fit()
coef_c = res_cont.params['skin_avg']
se_c = res_cont.bse['skin_avg']
rr_c = np.exp(coef_c)
ci_low_c = np.exp(coef_c - 1.96*se_c)
ci_high_c = np.exp(coef_c + 1.96*se_c)

# p value
pval_c = res_cont.pvalues['skin_avg']

# Also compute simple two-proportion test using red cards per game as rates: use poisson test for rate ratio with counts and exposures
# We'll compute rate ratio and p-value using statsmodels rate ratio test from statsmodels.stats.rates?

from statsmodels.stats.rates import test_poisson_2indep

dark = sub2[sub2['skin_dark']==1]
light = sub2[sub2['skin_dark']==0]

total_red_dark = dark['redCards'].sum()
exposure_dark = dark['games'].sum()

total_red_light = light['redCards'].sum()
exposure_light = light['games'].sum()

rate_test = test_poisson_2indep(
    total_red_dark,
    exposure_dark,
    total_red_light,
    exposure_light,
    method='score',
    compare='ratio',
)

# rate ratio and ci from test
rate_ratio = rate_test.ratio
try:
    ci = rate_test.conf_int()
    rr_ci_low, rr_ci_high = ci
except Exception:
    rr_ci_low, rr_ci_high = (np.nan, np.nan)

# p-value
pval_rate = rate_test.pvalue

# Save summary in a dict
out = {
    'missing_skin': int(missing_skin),
    'agg': agg,
    'poisson_rr': rr,
    'poisson_ci': (ci_low, ci_high),
    'poisson_p': pval,
    'poisson_cont_rr': rr_c,
    'poisson_cont_ci': (ci_low_c, ci_high_c),
    'poisson_cont_p': pval_c,
    'rate_ratio': rate_ratio,
    'rate_ci': (rr_ci_low, rr_ci_high),
    'rate_p': pval_rate,
    'total_red_dark': float(total_red_dark),
    'total_red_light': float(total_red_light),
    'total_games_dark': float(exposure_dark),
    'total_games_light': float(exposure_light),
}

# print summary
print('missing_skin', missing_skin)
print('agg')
print(agg)
print('poisson_rr', rr, 'ci', (ci_low, ci_high), 'p', pval)
print('poisson_cont_rr', rr_c, 'ci', (ci_low_c, ci_high_c), 'p', pval_c)
print('rate_ratio', rate_ratio, 'ci', (rr_ci_low, rr_ci_high), 'p', pval_rate)
print('totals dark red', total_red_dark, 'games', exposure_dark)
print('totals light red', total_red_light, 'games', exposure_light)

# save to csv for record? no
