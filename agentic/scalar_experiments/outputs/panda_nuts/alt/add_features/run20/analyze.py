import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure help and sex are categorical, normalize case
if 'help' in df.columns:
    df['help'] = df['help'].astype(str).str.strip().str.lower()
if 'sex' in df.columns:
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()

# Create efficiency metric: nuts opened per second
# Avoid division by zero
if 'nuts_opened' not in df.columns or 'seconds' not in df.columns:
    raise ValueError("Required columns missing for efficiency computation.")

df = df.copy()
# Replace zero seconds with NaN to avoid inf
secs = df['seconds'].replace(0, np.nan)
df['efficiency'] = df['nuts_opened'] / secs

# Drop rows with missing values in variables of interest
vars_of_interest = ['efficiency', 'age', 'sex', 'help']

df_model = df[vars_of_interest].dropna().copy()

# Encode categorical variables
# Use C() in formula

# Fit OLS model
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df_model).fit()

# Also try log-transformed efficiency to mitigate skew (add small constant)
# Ensure nonnegative
min_eff = df_model['efficiency'].min()
shift = 0.0
if min_eff <= 0:
    shift = abs(min_eff) + 1e-6

df_model['log_efficiency'] = np.log(df_model['efficiency'] + shift + 1e-6)
model_log = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=df_model).fit()

# Collect results
results = {
    'n': int(df_model.shape[0]),
    'efficiency_summary': df_model['efficiency'].describe().to_dict(),
    'ols': {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
    },
    'ols_log': {
        'params': model_log.params.to_dict(),
        'pvalues': model_log.pvalues.to_dict(),
        'rsquared': model_log.rsquared,
        'rsquared_adj': model_log.rsquared_adj,
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
