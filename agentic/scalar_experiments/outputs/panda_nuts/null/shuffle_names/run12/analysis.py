import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('panda_nuts.csv')

# Map columns to semantic variables based on info.json descriptions
# age_years appears to be in column 'hammer' (range 3-16, described as age in years)
# sex appears in column 'nuts_opened' (f/m)
# help appears in column 'seconds' (y/N)
# nuts opened count appears in column 'help'
# session duration seconds appears in column 'chimpanzee'

df = df.copy()

# Rename for clarity
df = df.rename(columns={
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'seconds': 'helped',
    'help': 'nuts_opened',
    'chimpanzee': 'session_seconds'
})

# Clean categorical variables
df['sex'] = df['sex'].astype('category')
df['hammer_type'] = df['hammer_type'].astype('category')
df['helped'] = df['helped'].str.upper().map({'Y': 'Y', 'N': 'N'})
df['helped'] = df['helped'].astype('category')

# Compute efficiency: nuts opened per minute (to make scale interpretable)
df['efficiency_per_min'] = df['nuts_opened'] / (df['session_seconds'] / 60.0)

# Also compute per second in case
df['efficiency_per_sec'] = df['nuts_opened'] / df['session_seconds']

print(df[['age_years','sex','helped','nuts_opened','session_seconds','efficiency_per_min']].head())

# Basic summary
summary = df[['age_years','nuts_opened','session_seconds','efficiency_per_min','efficiency_per_sec']].describe()
print(summary)

# OLS regression on efficiency per minute
model = smf.ols('efficiency_per_min ~ age_years + C(sex) + C(helped)', data=df).fit()
print(model.summary())

# Robust SEs (HC3)
model_hc3 = model.get_robustcov_results(cov_type='HC3')
print("\nHC3 robust summary")
print(model_hc3.summary())

# Also try log(1+efficiency) to reduce skew
df['log_eff'] = np.log1p(df['efficiency_per_min'])
model_log = smf.ols('log_eff ~ age_years + C(sex) + C(helped)', data=df).fit()
model_log_hc3 = model_log.get_robustcov_results(cov_type='HC3')
print("\nLog-efficiency HC3 summary")
print(model_log_hc3.summary())

# Save key results for later
params = pd.Series(model_hc3.params, index=model_hc3.model.exog_names)
pvalues = pd.Series(model_hc3.pvalues, index=model_hc3.model.exog_names)
log_params = pd.Series(model_log_hc3.params, index=model_log_hc3.model.exog_names)
log_pvalues = pd.Series(model_log_hc3.pvalues, index=model_log_hc3.model.exog_names)

results = {
    'n': len(df),
    'efficiency_per_min_mean': df['efficiency_per_min'].mean(),
    'efficiency_per_min_std': df['efficiency_per_min'].std(),
    'coeffs': params.to_dict(),
    'pvalues': pvalues.to_dict(),
    'log_coeffs': log_params.to_dict(),
    'log_pvalues': log_pvalues.to_dict(),
}
print("\nKEY_RESULTS")
print(results)
