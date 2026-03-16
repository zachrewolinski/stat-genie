import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Compute relative group size and relative contest location
# Relative size: focal - other
_df['rel_size'] = _df['n_focal'] - _df['n_other']
# Relative location: other distance - focal distance (positive => focal closer to its home range center)
_df['rel_dist'] = _df['dist_other'] - _df['dist_focal']

# Basic checks
print('n rows:', len(_df))
print('win rate:', _df['win'].mean())
print('\nrel_size summary:\n', _df['rel_size'].describe())
print('\nrel_dist summary:\n', _df['rel_dist'].describe())

# Logistic regression with both predictors
model = smf.logit('win ~ rel_size + rel_dist', data=_df).fit(disp=False)
print('\nLogit model summary:\n', model.summary())

# Odds ratios and 95% CI
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']
or_df = pd.DataFrame({
    'coef': params,
    'odds_ratio': np.exp(params),
    'or_2.5%': np.exp(conf['2.5%']),
    'or_97.5%': np.exp(conf['97.5%']),
    'p_value': model.pvalues
})
print('\nOdds ratios:\n', or_df)

# Also run model with standardized predictors for effect size comparison
_df['rel_size_z'] = (_df['rel_size'] - _df['rel_size'].mean()) / _df['rel_size'].std(ddof=0)
_df['rel_dist_z'] = (_df['rel_dist'] - _df['rel_dist'].mean()) / _df['rel_dist'].std(ddof=0)
model_z = smf.logit('win ~ rel_size_z + rel_dist_z', data=_df).fit(disp=False)
print('\nLogit model (z) summary:\n', model_z.summary())

# Simple bivariate checks
print('\nWin rate by rel_size positive vs not:')
_df['rel_size_pos'] = _df['rel_size'] > 0
print(_df.groupby('rel_size_pos')['win'].mean())

print('\nWin rate by rel_dist positive vs not:')
_df['rel_dist_pos'] = _df['rel_dist'] > 0
print(_df.groupby('rel_dist_pos')['win'].mean())

# Crosstab to see combined effects
ct = pd.crosstab(_df['rel_size_pos'], _df['rel_dist_pos'], values=_df['win'], aggfunc='mean')
print('\nWin rate by rel_size_pos x rel_dist_pos:\n', ct)

# Save key results for later use (optional)
print('\nModel AIC:', model.aic)
print('Pseudo R^2:', model.prsquared)
