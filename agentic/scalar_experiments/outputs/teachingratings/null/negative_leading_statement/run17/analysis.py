import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: drop rows with missing eval or beauty
cols_needed = ['eval', 'beauty']
df_clean = df.dropna(subset=cols_needed).copy()

# Simple correlation
corr = df_clean['eval'].corr(df_clean['beauty'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df_clean).fit(cov_type='HC3')

# Build controls: include available covariates
# Use all columns except identifiers/target; ensure categorical handling
# Potential controls: age, gender, minority, credits, division, native, tenure, students, allstudents
controls = ['age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'allstudents']

# Filter to controls that exist
controls = [c for c in controls if c in df_clean.columns]

# Build formula with categorical encoding
cat_vars = ['gender', 'minority', 'credits', 'division', 'native', 'tenure']
cat_vars = [c for c in cat_vars if c in controls]

# numerical controls
num_controls = [c for c in controls if c not in cat_vars]

terms = ['beauty']
terms += [f'C({c})' for c in cat_vars]
terms += num_controls

formula = 'eval ~ ' + ' + '.join(terms)
model_controls = smf.ols(formula, data=df_clean).fit(cov_type='HC3')

# Also include professor fixed effects? Might absorb beauty (beauty constant per prof) if multiple courses.
# But we'll skip because beauty varies by instructor; we can check number of courses per prof.

# Extract results
res = {
    'n': int(df_clean.shape[0]),
    'corr': corr,
    'simple_coef': model_simple.params.get('beauty', np.nan),
    'simple_p': model_simple.pvalues.get('beauty', np.nan),
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist() if 'beauty' in model_simple.params else [np.nan, np.nan],
    'controls_formula': formula,
    'controls_coef': model_controls.params.get('beauty', np.nan),
    'controls_p': model_controls.pvalues.get('beauty', np.nan),
    'controls_ci': model_controls.conf_int().loc['beauty'].tolist() if 'beauty' in model_controls.params else [np.nan, np.nan],
    'r2_simple': model_simple.rsquared,
    'r2_controls': model_controls.rsquared,
}

# Print summary
print('N', res['n'])
print('corr', res['corr'])
print('simple coef', res['simple_coef'], 'p', res['simple_p'], 'CI', res['simple_ci'], 'R2', res['r2_simple'])
print('controls formula', res['controls_formula'])
print('controls coef', res['controls_coef'], 'p', res['controls_p'], 'CI', res['controls_ci'], 'R2', res['r2_controls'])
