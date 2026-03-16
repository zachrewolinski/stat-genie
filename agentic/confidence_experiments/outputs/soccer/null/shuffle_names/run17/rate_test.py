import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.rates import test_poisson_2indep


df = pd.read_csv('soccer.csv')
df['mean_skin'] = df[['rater1','nExp']].mean(axis=1)
df['red_total'] = df['yellowCards'] + df['meanExp']
df['games_exposure'] = df['redCards']

analysis_df = df.dropna(subset=['mean_skin']).copy()
analysis_df = analysis_df[analysis_df['games_exposure']>0]

light = analysis_df[analysis_df['mean_skin'] <= 0.25]
dark = analysis_df[analysis_df['mean_skin'] >= 0.75]

rate_light = light['red_total'].sum() / light['games_exposure'].sum()
rate_dark = dark['red_total'].sum() / dark['games_exposure'].sum()

res = test_poisson_2indep(count1=dark['red_total'].sum(), exposure1=dark['games_exposure'].sum(),
                          count2=light['red_total'].sum(), exposure2=light['games_exposure'].sum(),
                          method='wald')
print('rate_light', rate_light, 'rate_dark', rate_dark)
print('rate_ratio', rate_dark/rate_light)
print('pvalue', res.pvalue)
print('confint', res.confint())
