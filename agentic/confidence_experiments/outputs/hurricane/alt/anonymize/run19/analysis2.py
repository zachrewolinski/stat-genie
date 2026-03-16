import pandas as pd
import numpy as np
import pingouin as pg
import statsmodels.api as sm

# Load data
cols = {
    'feature1':'id',
    'feature2':'year',
    'feature3':'name',
    'feature4':'fem_index',
    'feature5':'min_pressure',
    'feature6':'female_binary',
    'feature7':'category',
    'feature8':'deaths',
    'feature9':'damage_2013',
    'feature10':'years_since',
    'feature11':'source',
    'feature12':'fem_index_mturk',
    'feature13':'max_wind',
    'feature14':'damage_2015',
}

_df = pd.read_csv('hurricane.csv').rename(columns=cols)

# log1p deaths
_df['log1p_deaths'] = np.log1p(_df['deaths'])

# Partial correlations controlling for intensity variables
controls = ['min_pressure','category','max_wind']

pcorr1 = pg.partial_corr(data=_df, x='fem_index', y='log1p_deaths', covar=controls, method='pearson')
pcorr2 = pg.partial_corr(data=_df, x='fem_index_mturk', y='log1p_deaths', covar=controls, method='pearson')

print('Partial correlation fem_index vs log1p_deaths (controls):')
print(pcorr1)
print('Partial correlation fem_index_mturk vs log1p_deaths (controls):')
print(pcorr2)

# Spearman correlation (nonparametric) without controls
sp1 = pg.corr(_df['fem_index'], _df['deaths'], method='spearman')
sp2 = pg.corr(_df['fem_index_mturk'], _df['deaths'], method='spearman')
print('Spearman corr fem_index vs deaths')
print(sp1)
print('Spearman corr fem_index_mturk vs deaths')
print(sp2)

# Robust regression (RLM) on log1p deaths
X = _df[['fem_index','min_pressure','category','max_wind']]
X = sm.add_constant(X)
rlm = sm.RLM(_df['log1p_deaths'], X, M=sm.robust.norms.HuberT()).fit()
print('RLM log1p_deaths ~ fem_index + intensity')
print(rlm.summary())

# Check for interaction fem_index * category in RLM
_df['fem_x_cat'] = _df['fem_index'] * _df['category']
X2 = _df[['fem_index','category','fem_x_cat','min_pressure','max_wind']]
X2 = sm.add_constant(X2)
rlm2 = sm.RLM(_df['log1p_deaths'], X2, M=sm.robust.norms.HuberT()).fit()
print('RLM log1p_deaths ~ fem_index + category + interaction + intensity')
print(rlm2.summary())

# Save partial correlations
pcorr1.to_csv('partial_corr_fem_index.csv', index=False)
pcorr2.to_csv('partial_corr_fem_index_mturk.csv', index=False)
