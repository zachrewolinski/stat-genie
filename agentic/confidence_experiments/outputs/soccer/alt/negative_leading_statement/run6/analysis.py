import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint


df = pd.read_csv('soccer.csv')

# compute mean skin tone
skin = df[['rater1','rater2']].mean(axis=1, skipna=True)

df = df.assign(skin=skin)

# drop rows without skin
mask = df['skin'].notna()
df = df[mask].copy()

# aggregate to player level
# use playerShort as id
agg = df.groupby('playerShort').agg(
    total_games=('games','sum'),
    total_red=('redCards','sum'),
    skin=('skin','mean')
).reset_index()

# define categories
agg['skin_cat'] = np.select(
    [agg['skin'] <= 0.25, agg['skin'] >= 0.75],
    ['light','dark'],
    default='mid'
)

# filter to light/dark
ld = agg[agg['skin_cat'].isin(['light','dark'])].copy()
ld['dark'] = (ld['skin_cat']=='dark').astype(int)

# compute rates
ld['rate'] = ld['total_red'] / ld['total_games']

# summary stats
summary = ld.groupby('skin_cat').agg(
    players=('playerShort','count'),
    total_games=('total_games','sum'),
    total_red=('total_red','sum'),
    mean_rate=('rate','mean')
)

# rate ratio (dark / light) using aggregate rates
rate_dark = summary.loc['dark','total_red'] / summary.loc['dark','total_games']
rate_light = summary.loc['light','total_red'] / summary.loc['light','total_games']
rate_ratio = rate_dark / rate_light if rate_light>0 else np.nan

# Poisson regression with offset log(games)
X = sm.add_constant(ld['dark'])
model = sm.GLM(ld['total_red'], X, family=sm.families.Poisson(), offset=np.log(ld['total_games']))
res = model.fit(cov_type='HC0')

# Also continuous skin tone (per unit) as predictor
Xc = sm.add_constant(ld['skin'])
model_c = sm.GLM(ld['total_red'], Xc, family=sm.families.Poisson(), offset=np.log(ld['total_games']))
res_c = model_c.fit(cov_type='HC0')

# logistic for any red card
ld['any_red'] = (ld['total_red']>0).astype(int)
logit = sm.Logit(ld['any_red'], X)
logit_res = logit.fit(disp=False)

# compute proportions with any red
prop = ld.groupby('skin_cat')['any_red'].mean()

# confidence intervals for proportion
ci = {}
for cat in ['light','dark']:
    n = ld[ld['skin_cat']==cat]['any_red'].count()
    k = ld[ld['skin_cat']==cat]['any_red'].sum()
    ci[cat] = proportion_confint(k, n, method='wilson')

print('Players with photos (player-level):', len(agg))
print(summary)
print('Aggregate rate ratio (dark/light):', rate_ratio)
print('Poisson dark coef:', res.params['dark'], 'p:', res.pvalues['dark'])
print('Poisson dark rate ratio:', np.exp(res.params['dark']))
print('Poisson dark 95% CI:', np.exp(res.conf_int().loc['dark'].values))
print('Poisson continuous skin coef:', res_c.params['skin'], 'p:', res_c.pvalues['skin'])
print('Logit dark coef:', logit_res.params['dark'], 'p:', logit_res.pvalues['dark'])
print('Prop any red:', prop.to_dict())
print('Prop any red CI:', ci)
