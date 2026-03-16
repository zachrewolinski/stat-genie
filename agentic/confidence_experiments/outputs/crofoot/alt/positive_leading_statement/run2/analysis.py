import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Derived variables
_df['rel_group_size'] = _df['n_focal'] - _df['n_other']
# contest location: positive means contest closer to focal home range center
_df['rel_location'] = _df['dist_other'] - _df['dist_focal']

# Quick sanity
print(_df[['win','rel_group_size','rel_location']].describe())

# Logistic regression
# Use GLM binomial with robust SEs (HC1) due to small sample
model = smf.glm('win ~ rel_group_size + rel_location', data=_df, family=sm.families.Binomial())
result = model.fit(cov_type='HC1')
print(result.summary())

# Standardize predictors to compare effect sizes
_df['rel_group_size_z'] = (_df['rel_group_size'] - _df['rel_group_size'].mean()) / _df['rel_group_size'].std(ddof=0)
_df['rel_location_z'] = (_df['rel_location'] - _df['rel_location'].mean()) / _df['rel_location'].std(ddof=0)
model_z = smf.glm('win ~ rel_group_size_z + rel_location_z', data=_df, family=sm.families.Binomial())
result_z = model_z.fit(cov_type='HC1')
print(result_z.summary())

# Compute marginal effects at means
margeff = result.get_margeff(at='mean')
print(margeff.summary())

# Alternative: include dyad fixed effects? With 8 dyads, may be too many parameters.
# Fit model with dyad as categorical to check robustness
model_dyad = smf.glm('win ~ rel_group_size + rel_location + C(dyad)', data=_df, family=sm.families.Binomial())
result_dyad = model_dyad.fit()
print(result_dyad.summary())

# Likelihood ratio test for predictors vs intercept-only
model_null = smf.glm('win ~ 1', data=_df, family=sm.families.Binomial())
result_null = model_null.fit()
lr_stat = 2 * (result.llf - result_null.llf)
from scipy import stats
p_lr = stats.chi2.sf(lr_stat, df=2)
print('LR stat', lr_stat, 'p', p_lr)

# Save key results for later use
summary = {
    'n': int(_df.shape[0]),
    'coef': result.params.to_dict(),
    'pvalues': result.pvalues.to_dict(),
    'odds_ratio': np.exp(result.params).to_dict(),
    'coef_z': result_z.params.to_dict(),
    'pvalues_z': result_z.pvalues.to_dict(),
    'odds_ratio_z': np.exp(result_z.params).to_dict(),
    'lr_stat': float(lr_stat),
    'lr_p': float(p_lr),
}
print(json.dumps(summary, indent=2))
