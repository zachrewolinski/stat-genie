import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Rename columns based on info.json descriptions (shuffled names)
# age -> chimpanzee_id, hammer -> age_years, nuts_opened -> sex, sex -> hammer_type,
# help -> nuts_opened_count, chimpanzee -> seconds, seconds -> help_received
rename_map = {
    'age': 'chimpanzee_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened_count',
    'chimpanzee': 'seconds',
    'seconds': 'help_received'
}

df = df.rename(columns=rename_map)

# Clean/help variables
df['sex'] = df['sex'].astype('category')
df['help_received'] = df['help_received'].astype('category')

# Ensure numeric
for col in ['age_years', 'nuts_opened_count', 'seconds']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key values
analysis_df = df.dropna(subset=['age_years', 'sex', 'help_received', 'nuts_opened_count', 'seconds']).copy()

# Efficiency as rate (nuts per second)
analysis_df['rate'] = analysis_df['nuts_opened_count'] / analysis_df['seconds']

# Poisson GLM with offset for seconds to model rate
analysis_df['log_seconds'] = np.log(analysis_df['seconds'])

model_poisson = smf.glm(
    formula='nuts_opened_count ~ age_years + sex + help_received',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_seconds']
).fit(cov_type='HC0')

# Also fit linear model on log(rate) for robustness (add small epsilon)
analysis_df['log_rate'] = np.log(analysis_df['rate'] + 1e-9)
model_ols = smf.ols('log_rate ~ age_years + sex + help_received', data=analysis_df).fit(cov_type='HC0')

# Summaries
poisson_params = model_poisson.params
poisson_pvals = model_poisson.pvalues

ols_params = model_ols.params
ols_pvals = model_ols.pvalues

# Collect results
results = {
    'n': len(analysis_df),
    'poisson_params': poisson_params.to_dict(),
    'poisson_pvals': poisson_pvals.to_dict(),
    'ols_params': ols_params.to_dict(),
    'ols_pvals': ols_pvals.to_dict(),
}

# Save a lightweight JSON for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Rows:', len(analysis_df))
print('Poisson p-values:')
print(poisson_pvals)
print('\nOLS p-values:')
print(ols_pvals)
