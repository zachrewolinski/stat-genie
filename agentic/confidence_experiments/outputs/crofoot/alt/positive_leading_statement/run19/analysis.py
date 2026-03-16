import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Create relative metrics
_df['rel_size'] = _df['n_focal'] - _df['n_other']
# Positive rel_loc means focal is closer to its home range center than other (focal more at home)
_df['rel_loc'] = _df['dist_other'] - _df['dist_focal']

# Basic checks
print('rows', len(_df))
print(_df[['win','rel_size','rel_loc']].describe())

# Logistic regression
model = smf.glm('win ~ rel_size + rel_loc', data=_df, family=sm.families.Binomial()).fit()
print(model.summary())

# Also check with standardized predictors for effect size comparison
_df['rel_size_z'] = (_df['rel_size'] - _df['rel_size'].mean()) / _df['rel_size'].std(ddof=0)
_df['rel_loc_z'] = (_df['rel_loc'] - _df['rel_loc'].mean()) / _df['rel_loc'].std(ddof=0)
model_z = smf.glm('win ~ rel_size_z + rel_loc_z', data=_df, family=sm.families.Binomial()).fit()
print(model_z.summary())

# Odds ratios and CIs
params = model.params
conf = model.conf_int()
ors = np.exp(params)
ci_low = np.exp(conf[0])
ci_high = np.exp(conf[1])
print('\nOdds ratios (per 1 unit):')
for name in params.index:
    print(name, ors[name], ci_low[name], ci_high[name], 'p', model.pvalues[name])

# Evaluate both predictors in a model with interaction (exploratory)
model_int = smf.glm('win ~ rel_size + rel_loc + rel_size:rel_loc', data=_df, family=sm.families.Binomial()).fit()
print(model_int.summary())

# Simple bivariate tests
# Compare win rate by sign of rel_size and rel_loc
_df['rel_size_pos'] = _df['rel_size'] > 0
_df['rel_loc_pos'] = _df['rel_loc'] > 0
print('win rate rel_size_pos', _df.groupby('rel_size_pos')['win'].mean())
print('win rate rel_loc_pos', _df.groupby('rel_loc_pos')['win'].mean())

# Chi-square for those binary splits
from scipy import stats
for col in ['rel_size_pos','rel_loc_pos']:
    table = pd.crosstab(_df[col], _df['win'])
    chi2, p, dof, exp = stats.chi2_contingency(table)
    print(col, 'chi2', chi2, 'p', p, 'table\n', table)

