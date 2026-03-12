import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic info
print('rows', _df.shape[0])
print('cols', _df.shape[1])

# Correlation between beauty and eval
corr = _df['beauty'].corr(_df['eval'])
print('corr_beauty_eval', corr)

# Simple regression
model_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC3')
print('\nSimple regression (HC3):')
print(model_simple.summary().tables[1])

# Controls model
# Convert categorical columns to categories for patsy
cat_cols = ['minority','gender','credits','division','native','tenure']
for c in cat_cols:
    _df[c] = _df[c].astype('category')

formula_controls = (
    'eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + '
    'C(credits) + C(division) + C(native) + C(tenure)'
)
model_controls = smf.ols(formula_controls, data=_df).fit(cov_type='HC3')
print('\nControls regression (HC3):')
print(model_controls.summary().tables[1])

# Standardized effect: beta for beauty
# Compute standardized coefficients for beauty in simple and controls
# Standardize beauty and eval; then estimate coefficient
_df_std = _df.copy()
_df_std['beauty_z'] = (_df_std['beauty'] - _df_std['beauty'].mean()) / _df_std['beauty'].std(ddof=0)
_df_std['eval_z'] = (_df_std['eval'] - _df_std['eval'].mean()) / _df_std['eval'].std(ddof=0)
model_std = smf.ols('eval_z ~ beauty_z', data=_df_std).fit(cov_type='HC3')
print('\nStandardized simple regression (HC3):')
print(model_std.summary().tables[1])

# Save key results for reference
results = {
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'std_coef': model_std.params['beauty_z'],
    'n': len(_df)
}

print('\nKEY_RESULTS')
for k,v in results.items():
    print(k, v)
