import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Basic cleaning
# Normalize help values (y/N) maybe case-insensitive
# Ensure sex and help as category
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Compute efficiency as nuts opened per second
# Avoid division by zero (seconds min is 2.5 per metadata)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Model: efficiency ~ age + sex + help
# Use categorical coding for sex and help
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Also model nuts_opened with offset log(seconds) using Poisson (count per time)
# Add small epsilon for log(seconds) if needed
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                  family=sm.families.Poisson(),
                  offset=np.log(df['seconds'])).fit(cov_type='HC3')

# Extract key stats
results = {
    'n': int(df.shape[0]),
    'efficiency_mean': float(df['efficiency'].mean()),
    'efficiency_std': float(df['efficiency'].std()),
    'ols_params': model.params.to_dict(),
    'ols_pvalues': model.pvalues.to_dict(),
    'ols_r2': float(model.rsquared),
    'poisson_params': poisson.params.to_dict(),
    'poisson_pvalues': poisson.pvalues.to_dict(),
    'poisson_deviance': float(poisson.deviance),
}

# Save results for inspection
pd.DataFrame({'term': list(results['ols_params'].keys()),
              'coef': list(results['ols_params'].values()),
              'pvalue': [results['ols_pvalues'][k] for k in results['ols_params']]}).to_csv('ols_results.csv', index=False)

pd.DataFrame({'term': list(results['poisson_params'].keys()),
              'coef': list(results['poisson_params'].values()),
              'pvalue': [results['poisson_pvalues'][k] for k in results['poisson_params']]}).to_csv('poisson_results.csv', index=False)

# Print a concise summary
print('N', results['n'])
print('efficiency mean', results['efficiency_mean'])
print('efficiency std', results['efficiency_std'])
print('OLS R2', results['ols_r2'])
print('OLS params', results['ols_params'])
print('OLS pvalues', results['ols_pvalues'])
print('Poisson params', results['poisson_params'])
print('Poisson pvalues', results['poisson_pvalues'])
