import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'panda_nuts.csv'
info_path = 'info.json'

with open(info_path, 'r') as f:
    info = json.load(f)

_df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure expected columns
expected_cols = set(info['data_desc']['field_names'])
missing = expected_cols - set(_df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Efficiency: nuts opened per second (rate)
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Encode categorical variables
# statsmodels formula with C() handles categories

# OLS on efficiency
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')

# GLM Poisson on counts with log(seconds) offset (rate model)
_df['log_seconds'] = np.log(_df['seconds'])
poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=_df,
                        family=sm.families.Poisson(),
                        offset=_df['log_seconds']).fit(cov_type='HC3')

# Summaries for inference
ols_pvalues = ols_model.pvalues
poisson_pvalues = poisson_model.pvalues

# Collect key effects
results = {
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_pvalues.to_dict(),
    'ols_r2': ols_model.rsquared,
    'poisson_params': poisson_model.params.to_dict(),
    'poisson_pvalues': poisson_pvalues.to_dict(),
    'poisson_llf': poisson_model.llf,
    'n': int(len(_df)),
}

# Save a small json for reference
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('OLS p-values')
print(ols_pvalues)
print('\nPoisson p-values')
print(poisson_pvalues)
print('\nOLS params')
print(ols_model.params)
print('\nPoisson params')
print(poisson_model.params)
print('\nOLS R2:', ols_model.rsquared)
