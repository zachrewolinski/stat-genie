import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns based on observed values and metadata notes
# age: numeric 1-22
# sex: 'f'/'m' stored in column nuts_opened
# help_received: 'y'/'N' stored in column seconds
# nuts_opened_count: numeric stored in help
# duration_seconds: numeric stored in chimpanzee

# Compute efficiency as nuts opened per second
# Avoid division by zero (none expected but guard just in case)
df = df.copy()

df['efficiency'] = df['help'] / df['chimpanzee']

# Recode categorical variables
# standardize help received to Yes/No
help_map = {'y': 'yes', 'Y': 'yes', 'N': 'no', 'n': 'no'}
df['help_received'] = df['seconds'].map(help_map)

# sex from nuts_opened column
sex_map = {'f': 'f', 'm': 'm'}
df['sex_chimp'] = df['nuts_opened'].map(sex_map)

# Drop rows with missing mapped values
analysis_df = df.dropna(subset=['efficiency', 'age', 'help_received', 'sex_chimp']).copy()

# Fit OLS with categorical predictors
model = smf.ols('efficiency ~ age + C(sex_chimp) + C(help_received)', data=analysis_df).fit()

# Also test a log-transformed efficiency to reduce skew (add small constant)
analysis_df['log_eff'] = np.log(analysis_df['efficiency'] + 1e-6)
log_model = smf.ols('log_eff ~ age + C(sex_chimp) + C(help_received)', data=analysis_df).fit()

# Collect key results
results = {
    'n': int(analysis_df.shape[0]),
    'efficiency_mean': float(analysis_df['efficiency'].mean()),
    'efficiency_median': float(analysis_df['efficiency'].median()),
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_rsquared': float(model.rsquared),
    'log_model_params': log_model.params.to_dict(),
    'log_model_pvalues': log_model.pvalues.to_dict(),
    'log_model_rsquared': float(log_model.rsquared),
}

# Save results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
