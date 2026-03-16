import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('soccer.csv')
df['skin_mean'] = df[['rater1','rater2']].mean(axis=1)
base = df[(df['skin_mean'].notna()) & (df['games'] > 0)].copy()
cont_df = base[['redCards','skin_mean','games']].dropna().copy()
cont_df['log_games'] = np.log(cont_df['games'])
model = sm.GLM(cont_df['redCards'], sm.add_constant(cont_df['skin_mean']),
               family=sm.families.Poisson(), offset=cont_df['log_games']).fit(cov_type='HC1')
coef = model.params['skin_mean']
ci_low, ci_high = model.conf_int().loc['skin_mean']
print('coef', coef)
print('p', model.pvalues['skin_mean'])
print('ci', ci_low, ci_high)
