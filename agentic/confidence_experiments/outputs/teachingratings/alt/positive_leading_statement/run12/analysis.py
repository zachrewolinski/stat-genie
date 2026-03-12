import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure categorical columns are treated as such
cat_cols = ['minority', 'gender', 'credits', 'division', 'native', 'tenure']
for c in cat_cols:
    if c in _df.columns:
        _df[c] = _df[c].astype('category')

# Simple correlation
corr = _df[['beauty', 'eval']].corr().iloc[0, 1]

# Simple regression
model_simple = smf.ols('eval ~ beauty', data=_df).fit()

# Add controls (course + instructor characteristics)
# Use log of class sizes to reduce skew
_df = _df.copy()
_df['log_students'] = np.log(_df['students'])
_df['log_allstudents'] = np.log(_df['allstudents'])

controls = (
    'beauty + age + C(gender) + C(minority) + C(native) + C(tenure) '
    '+ C(division) + C(credits) + log_students + log_allstudents'
)
model_controls = smf.ols(f'eval ~ {controls}', data=_df).fit()

# Alternative control set without log_allstudents to reduce collinearity
model_controls2 = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) '
    '+ C(division) + C(credits) + log_students',
    data=_df,
).fit()

# Summaries
out = {
    'n': len(_df),
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist(),
    'controls2_coef': model_controls2.params['beauty'],
    'controls2_p': model_controls2.pvalues['beauty'],
    'controls2_ci': model_controls2.conf_int().loc['beauty'].tolist(),
    'r2_simple': model_simple.rsquared,
    'r2_controls': model_controls.rsquared,
    'r2_controls2': model_controls2.rsquared,
}

print(out)
