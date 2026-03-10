import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Create predictors
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_dist'] = _df['dist_other'] - _df['dist_focal']  # positive if focal closer to its center than other is to theirs

# Also compute ratio for robustness
_df['size_ratio'] = _df['n_focal'] / _df['n_other']

# Basic descriptive stats
print('n rows', len(_df))
print(_df[['win','rel_size','rel_dist','size_ratio']].describe())

# Logistic regression with rel_size and rel_dist
model1 = smf.logit('win ~ rel_size + rel_dist', data=_df).fit(disp=False)
print('\nModel1: win ~ rel_size + rel_dist')
print(model1.summary())

# Logistic regression with size_ratio and rel_dist
model2 = smf.logit('win ~ size_ratio + rel_dist', data=_df).fit(disp=False)
print('\nModel2: win ~ size_ratio + rel_dist')
print(model2.summary())

# Simple univariate tests
model_size = smf.logit('win ~ rel_size', data=_df).fit(disp=False)
model_dist = smf.logit('win ~ rel_dist', data=_df).fit(disp=False)
print('\nModel_size: win ~ rel_size')
print(model_size.summary())
print('\nModel_dist: win ~ rel_dist')
print(model_dist.summary())

# Predicted probabilities at +/- 1 sd for rel_size and rel_dist using model1
sd_size = _df['rel_size'].std()
sd_dist = _df['rel_dist'].std()

params = model1.params

for name, delta_size, delta_dist in [
    ('low size, low dist', -sd_size, -sd_dist),
    ('high size, low dist', sd_size, -sd_dist),
    ('low size, high dist', -sd_size, sd_dist),
    ('high size, high dist', sd_size, sd_dist),
]:
    lin = params['Intercept'] + params['rel_size']*delta_size + params['rel_dist']*delta_dist
    prob = 1/(1+np.exp(-lin))
    print(f'pred {name}: {prob:.3f}')

# Save key results
results = {
    'model1_params': model1.params.to_dict(),
    'model1_pvalues': model1.pvalues.to_dict(),
    'model1_n': int(model1.nobs),
    'model1_pseudo_r2': model1.prsquared,
    'model2_params': model2.params.to_dict(),
    'model2_pvalues': model2.pvalues.to_dict(),
    'model_size_pvalues': model_size.pvalues.to_dict(),
    'model_dist_pvalues': model_dist.pvalues.to_dict(),
}

pd.Series(results).to_json('analysis_results.json')
