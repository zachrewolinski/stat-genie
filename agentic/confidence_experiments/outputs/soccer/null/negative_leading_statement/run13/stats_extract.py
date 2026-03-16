import numpy as np
import pandas as pd
import statsmodels.api as sm

# Use row-level skin_mean as in analysis_row

df = pd.read_csv('soccer.csv')
df['skin_mean'] = df[['rater1','rater2']].mean(axis=1)
base = df[(df['skin_mean'].notna()) & (df['games'] > 0)].copy()
base['skin_group'] = np.where(base['skin_mean'] <= 0.25, 'light', np.where(base['skin_mean'] >= 0.75, 'dark', 'mid'))

summary = (
    base.groupby('skin_group')
    .agg(dyads=('playerShort','size'),
         players=('playerShort','nunique'),
         total_games=('games','sum'),
         total_red=('redCards','sum'))
    .assign(red_per_game=lambda x: x['total_red'] / x['total_games'])
)

bin_df = base[base['skin_group'].isin(['dark','light'])].copy()
bin_df['log_games'] = np.log(bin_df['games'])
bin_df['dark'] = (bin_df['skin_group'] == 'dark').astype(int)
model = sm.GLM(bin_df['redCards'], sm.add_constant(bin_df['dark']),
               family=sm.families.Poisson(), offset=bin_df['log_games']).fit(cov_type='HC1')
coef = model.params['dark']
se = model.bse['dark']
ci_low, ci_high = model.conf_int().loc['dark']

irr = float(np.exp(coef))
irr_low = float(np.exp(ci_low))
irr_high = float(np.exp(ci_high))

print('summary')
print(summary)
print('coef', coef)
print('p', model.pvalues['dark'])
print('irr', irr, 'irr_ci', irr_low, irr_high)
print('rate_light', summary.loc['light','red_per_game'])
print('rate_dark', summary.loc['dark','red_per_game'])
print('rate_diff', summary.loc['dark','red_per_game'] - summary.loc['light','red_per_game'])
