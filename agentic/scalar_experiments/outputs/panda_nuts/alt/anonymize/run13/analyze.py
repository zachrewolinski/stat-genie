import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path(__file__).with_name('panda_nuts.csv')

df = pd.read_csv(DATA_PATH)

# Rename for clarity
cols = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer_type',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}

df = df.rename(columns=cols)

# Basic cleaning
# Ensure categorical types
for c in ['sex', 'hammer_type', 'help']:
    df[c] = df[c].astype('category')

# Efficiency: nuts opened per minute (consistent scale)
df['efficiency_per_min'] = df['nuts_opened'] / (df['duration_sec'] / 60.0)

# Drop any infinite or missing values
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['efficiency_per_min', 'age', 'sex', 'help'])

# Encode help to y/n if mixed case
# Keep as category but standardize
if df['help'].dtype.name == 'category':
    df['help'] = df['help'].cat.remove_unused_categories()

# Descriptive stats
summary = df[['efficiency_per_min', 'age']].describe().T

# Model: efficiency ~ age + sex + help
# Use OLS with robust standard errors (HC3)
model = smf.ols('efficiency_per_min ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Also check log-efficiency to reduce skew
# Add small constant to avoid log(0)
model_log = smf.ols('np.log(efficiency_per_min + 1e-6) ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

print('N', len(df))
print(summary)
print('\nOLS robust summary')
print(model.summary())
print('\nLog-OLS robust summary')
print(model_log.summary())

# Group means for help and sex
print('\nGroup means by help:')
print(df.groupby('help')['efficiency_per_min'].mean())
print('\nGroup means by sex:')
print(df.groupby('sex')['efficiency_per_min'].mean())

# Partial effects (marginal):
params = model.params
pvals = model.pvalues
print('\nParams (OLS):')
print(params)
print('\nP-values (OLS):')
print(pvals)

params_log = model_log.params
pvals_log = model_log.pvalues
print('\nParams (Log OLS):')
print(params_log)
print('\nP-values (Log OLS):')
print(pvals_log)
