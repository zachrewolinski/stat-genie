import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic prep
# Log-transform deaths for stability
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Standardize some predictors for comparability (optional)
for col in ['masfem', 'wind', 'min', 'category']:
    _df[f'z_{col}'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

results = {}

# Correlations
results['corr_masfem_deaths_pearson'] = _df['masfem'].corr(_df['alldeaths'], method='pearson')
results['corr_masfem_deaths_spearman'] = _df['masfem'].corr(_df['alldeaths'], method='spearman')
results['corr_masfem_log_deaths_pearson'] = _df['masfem'].corr(_df['log_deaths'], method='pearson')

# Simple bivariate regression
model_simple = smf.ols('log_deaths ~ masfem', data=_df).fit()
results['simple_coef'] = model_simple.params['masfem']
results['simple_pval'] = model_simple.pvalues['masfem']

# Multivariate regression with storm intensity controls
model_controls = smf.ols('log_deaths ~ masfem + wind + min + category', data=_df).fit()
results['controls_coef'] = model_controls.params['masfem']
results['controls_pval'] = model_controls.pvalues['masfem']

# Regression with standardized predictors for effect size
model_controls_z = smf.ols('log_deaths ~ z_masfem + z_wind + z_min + z_category', data=_df).fit()
results['controls_z_coef'] = model_controls_z.params['z_masfem']
results['controls_z_pval'] = model_controls_z.pvalues['z_masfem']

# Alternative: gender_mf binary
model_gender = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=_df).fit()
results['gender_coef'] = model_gender.params['gender_mf']
results['gender_pval'] = model_gender.pvalues['gender_mf']

# Save results to a json file for later inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print concise summary
print(json.dumps(results, indent=2))
