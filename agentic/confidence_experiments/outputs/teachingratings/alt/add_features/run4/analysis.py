import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic info
n = len(df)

# Check missing values in key columns
key_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students']
missing = df[key_cols].isna().sum()

# Simple correlation
corr = df['beauty'].corr(df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# Controls (categorical as factors)
# Use students (participants) and age as numeric.
model_controls = smf.ols(
    'eval ~ beauty + age + students + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)',
    data=df
).fit()

# Clustered SE by prof (if available) for robustness
model_controls_cluster = model_controls.get_robustcov_results(cov_type='cluster', groups=df['prof'])

# Compute standardized effect (beta) for beauty in simple model
# Standardize beauty and eval
beauty_std = df['beauty'].std()
eval_std = df['eval'].std()
std_beta_simple = model_simple.params['beauty'] * beauty_std / eval_std

# Standardized effect for controls model
std_beta_controls = model_controls.params['beauty'] * beauty_std / eval_std

# Print key results
print('N', n)
print('Missing', missing.to_dict())
print('Correlation beauty-eval', corr)

print('Simple OLS beauty coef', model_simple.params['beauty'], 'p', model_simple.pvalues['beauty'])
print('Simple OLS R2', model_simple.rsquared)
print('Std beta simple', std_beta_simple)

print('Controls OLS beauty coef', model_controls.params['beauty'], 'p', model_controls.pvalues['beauty'])
print('Controls OLS R2', model_controls.rsquared)
print('Std beta controls', std_beta_controls)

print('Controls cluster-robust beauty coef', model_controls_cluster.params[model_controls_cluster.model.exog_names.index('beauty')],
      'p', model_controls_cluster.pvalues[model_controls_cluster.model.exog_names.index('beauty')])

# Save summary stats for later
summary = {
    'n': n,
    'corr': corr,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_r2': model_controls.rsquared,
    'controls_cluster_p': model_controls_cluster.pvalues[model_controls_cluster.model.exog_names.index('beauty')],
    'controls_cluster_coef': model_controls_cluster.params[model_controls_cluster.model.exog_names.index('beauty')],
    'std_beta_simple': std_beta_simple,
    'std_beta_controls': std_beta_controls,
}

pd.Series(summary).to_json('analysis_summary.json')
