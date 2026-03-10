import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('soccer.csv')

# Identify columns (based on data inspection)
# rater1 and nExp appear to be 5-level skin ratings in [0,1]
_df['skin_mean'] = _df[['rater1','nExp']].mean(axis=1)

# games column: redCards
_df['games'] = _df['redCards']

# red cards: direct red + second-yellow
_df['red_direct'] = _df['yellowCards']
_df['red_second'] = _df['meanExp']
_df['red_total'] = _df['red_direct'] + _df['red_second']

# drop rows with missing skin
df = _df.dropna(subset=['skin_mean','games','red_total'])

# Basic rates by skin_mean categories (rounded to nearest 0.25 for readability)
df['skin_round'] = df['skin_mean'].round(2)
rate_by_skin = df.groupby('skin_round').apply(lambda g: pd.Series({
    'n': len(g),
    'games': g['games'].sum(),
    'red_total': g['red_total'].sum(),
    'red_rate': g['red_total'].sum()/g['games'].sum(),
})).reset_index().sort_values('skin_round')

print('rate_by_skin (first 10)')
print(rate_by_skin.head(10))
print('rate_by_skin (last 10)')
print(rate_by_skin.tail(10))

# Dark vs light using thresholds on skin_mean
light = df[df['skin_mean'] <= 0.25]
dark = df[df['skin_mean'] >= 0.75]
print('light n', len(light), 'dark n', len(dark))
if len(light) > 0 and len(dark) > 0:
    light_rate = light['red_total'].sum()/light['games'].sum()
    dark_rate = dark['red_total'].sum()/dark['games'].sum()
    print('light_rate', light_rate, 'dark_rate', dark_rate, 'rate_ratio', dark_rate/light_rate if light_rate>0 else np.nan)

# Poisson regression with offset log(games)
# Use skin_mean as predictor (0-1)
X = sm.add_constant(df['skin_mean'])
model = sm.GLM(df['red_total'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
res = model.fit(cov_type='HC1')
print(res.summary())

# Negative binomial for robustness
nb_model = sm.GLM(df['red_total'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(df['games']))
nb_res = nb_model.fit(cov_type='HC1')
print(nb_res.summary())

# Also check using rater1 alone and nExp alone
for skin_col in ['rater1','nExp']:
    d2 = _df.dropna(subset=[skin_col,'games','red_total']).copy()
    X2 = sm.add_constant(d2[skin_col])
    m2 = sm.GLM(d2['red_total'], X2, family=sm.families.Poisson(), offset=np.log(d2['games']))
    r2 = m2.fit(cov_type='HC1')
    print('poisson', skin_col, r2.params, r2.pvalues)

