import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('soccer.csv')
_df['skin_mean'] = _df[['rater1','nExp']].mean(axis=1)
_df['games'] = _df['redCards']
_df['red_direct'] = _df['yellowCards']

df = _df.dropna(subset=['skin_mean','games','red_direct'])

X = sm.add_constant(df['skin_mean'])
model = sm.GLM(df['red_direct'], X, family=sm.families.Poisson(), offset=np.log(df['games']))
res = model.fit(cov_type='HC1')
print(res.summary())

light = df[df['skin_mean'] <= 0.25]
dark = df[df['skin_mean'] >= 0.75]
print('light n', len(light), 'dark n', len(dark))
if len(light) > 0 and len(dark) > 0:
    light_rate = light['red_direct'].sum()/light['games'].sum()
    dark_rate = dark['red_direct'].sum()/dark['games'].sum()
    print('light_rate', light_rate, 'dark_rate', dark_rate, 'rate_ratio', dark_rate/light_rate if light_rate>0 else np.nan)

