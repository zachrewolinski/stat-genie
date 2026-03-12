import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Derived variables
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['size_ratio'] = _df['n_focal'] / _df['n_other']
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']  # positive: contest closer to focal's range center
_df['loc_adv_scaled'] = (_df['loc_adv'] - _df['loc_adv'].mean()) / _df['loc_adv'].std(ddof=0)
_df['rel_size_scaled'] = (_df['rel_size'] - _df['rel_size'].mean()) / _df['rel_size'].std(ddof=0)

print(_df[['win','rel_size','size_ratio','loc_adv']].describe())

# Logistic regression with rel_size and loc_adv
# Use standardized predictors to compare
model = smf.logit('win ~ rel_size_scaled + loc_adv_scaled', data=_df).fit(disp=False)
print('\nLogit: win ~ rel_size_scaled + loc_adv_scaled')
print(model.summary())

# Alternative with size_ratio
model_ratio = smf.logit('win ~ size_ratio + loc_adv_scaled', data=_df).fit(disp=False)
print('\nLogit: win ~ size_ratio + loc_adv_scaled')
print(model_ratio.summary())

# Simple group comparisons
_df['size_adv'] = (_df['rel_size'] > 0).astype(int)
_df['loc_adv_bin'] = (_df['loc_adv'] > 0).astype(int)

summary = _df.groupby('size_adv')['win'].agg(['mean','count'])
print('\nWin rate by size_adv (1 if focal larger):')
print(summary)

summary_loc = _df.groupby('loc_adv_bin')['win'].agg(['mean','count'])
print('\nWin rate by loc_adv_bin (1 if closer to focal center):')
print(summary_loc)

# Two-proportion tests (approx) using statsmodels
from statsmodels.stats.proportion import proportions_ztest

# size_adv
counts = _df.groupby('size_adv')['win'].sum()
ns = _df.groupby('size_adv')['win'].count()
if len(counts)==2:
    z,p = proportions_ztest(counts.values, ns.values)
    print('\nTwo-proportion z-test size_adv: z=%.3f p=%.4f' % (z,p))

# loc_adv_bin
counts = _df.groupby('loc_adv_bin')['win'].sum()
ns = _df.groupby('loc_adv_bin')['win'].count()
if len(counts)==2:
    z,p = proportions_ztest(counts.values, ns.values)
    print('Two-proportion z-test loc_adv_bin: z=%.3f p=%.4f' % (z,p))

# Combined additive effects logistic with both binary indicators
model_bin = smf.logit('win ~ size_adv + loc_adv_bin', data=_df).fit(disp=False)
print('\nLogit: win ~ size_adv + loc_adv_bin')
print(model_bin.summary())

# Interactions
model_inter = smf.logit('win ~ rel_size_scaled * loc_adv_scaled', data=_df).fit(disp=False)
print('\nLogit: win ~ rel_size_scaled * loc_adv_scaled')
print(model_inter.summary())

# McFadden R2 for main model
llf = model.llf
llnull = model.llnull
r2 = 1 - llf/llnull
print('\nMcFadden R2 (main model): %.4f' % r2)
