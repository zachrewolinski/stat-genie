import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Create derived variables
_df['size_diff'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['dist_diff'] = _df['dist_focal'] - _df['dist_other']
_df['dist_ratio'] = _df['dist_focal'] / _df['dist_other']

# Basic checks
print('Rows:', len(_df))
print('Win rate:', _df['win'].mean())
print('Size diff summary:', _df['size_diff'].describe())
print('Dist diff summary:', _df['dist_diff'].describe())

# Logistic regression: win ~ size_diff + dist_diff
model1 = smf.logit('win ~ size_diff + dist_diff', data=_df).fit(disp=False)
print('\nLogit win ~ size_diff + dist_diff')
print(model1.summary())

# Logistic regression with size_ratio and dist_ratio (log transform ratio for stability)
_df['log_size_ratio'] = np.log(_df['size_ratio'])
_df['log_dist_ratio'] = np.log(_df['dist_ratio'])
model2 = smf.logit('win ~ log_size_ratio + log_dist_ratio', data=_df).fit(disp=False)
print('\nLogit win ~ log_size_ratio + log_dist_ratio')
print(model2.summary())

# Add both raw and interaction as sensitivity
model3 = smf.logit('win ~ size_diff + dist_diff + size_diff:dist_diff', data=_df).fit(disp=False)
print('\nLogit win ~ size_diff + dist_diff + interaction')
print(model3.summary())

# Alternative: use dist_focal and dist_other separately
model4 = smf.logit('win ~ size_diff + dist_focal + dist_other', data=_df).fit(disp=False)
print('\nLogit win ~ size_diff + dist_focal + dist_other')
print(model4.summary())

# Correlation checks
print('\nCorrelation win with size_diff:', _df['win'].corr(_df['size_diff']))
print('Correlation win with dist_diff:', _df['win'].corr(_df['dist_diff']))

