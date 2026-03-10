import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Rename key columns for readability
rename_map = {
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'eval_score',
    'feature8': 'division',
    'feature9': 'native_english',
    'feature10': 'tenure_track',
    'feature11': 'n_eval',
    'feature12': 'n_enroll',
    'feature13': 'instructor_id',
}

_df = _df.rename(columns=rename_map)

# Basic correlation
corr = _df['beauty'].corr(_df['eval_score'])

# Simple OLS: eval_score ~ beauty
model_simple = smf.ols('eval_score ~ beauty', data=_df).fit(cov_type='HC3')

# Multiple OLS with controls
# Use categorical controls for binary factors
model_controls = smf.ols(
    'eval_score ~ beauty + age + C(gender) + C(minority) + C(single_credit) + C(division) '
    '+ C(native_english) + C(tenure_track) + n_eval + n_enroll',
    data=_df
).fit(cov_type='HC3')

# Instructor fixed effects (robustness)
model_fe = smf.ols(
    'eval_score ~ beauty + age + C(gender) + C(minority) + C(single_credit) + C(division) '
    '+ C(native_english) + C(tenure_track) + n_eval + n_enroll + C(instructor_id)',
    data=_df
).fit(cov_type='HC3')

print('N:', len(_df))
print('Corr(beauty, eval_score):', corr)
print('\nSimple OLS (HC3):')
print(model_simple.params[['beauty']])
print(model_simple.pvalues[['beauty']])
print('\nControls OLS (HC3):')
print(model_controls.params[['beauty']])
print(model_controls.pvalues[['beauty']])
print('\nControls + instructor FE (HC3):')
print(model_fe.params[['beauty']])
print(model_fe.pvalues[['beauty']])

# Effect size interpretation: change in eval_score for 1 SD in beauty
beauty_sd = _df['beauty'].std()
coef = model_controls.params['beauty']
print('\nBeauty SD:', beauty_sd)
print('Eval change per 1 SD beauty (controls):', coef * beauty_sd)
