import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns based on observed values
# age: numeric age in years
# nuts_opened: sex (m/f)
# seconds: help received (y/N)
# help: count of nuts opened
# chimpanzee: session duration in seconds

# Clean/encode variables
sex = df['nuts_opened'].astype(str).str.strip().str.lower()
# Normalize to 'm'/'f'
sex = sex.replace({'male': 'm', 'female': 'f'})

help_received = df['seconds'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
# If any missing due to unexpected values, fill with NaN

analysis_df = pd.DataFrame({
    'age': df['age'].astype(float),
    'sex': sex,
    'help_received': help_received,
    'nuts_opened_count': df['help'].astype(float),
    'session_seconds': df['chimpanzee'].astype(float),
})

# Drop rows with missing/invalid values
analysis_df = analysis_df.dropna()
analysis_df = analysis_df[analysis_df['session_seconds'] > 0]

# Compute rate (nuts per second) for descriptive stats
analysis_df['rate_per_sec'] = analysis_df['nuts_opened_count'] / analysis_df['session_seconds']

# Poisson regression with log offset for session time
# Model rate of nuts opened per second as function of age, sex, and help_received
formula = 'nuts_opened_count ~ age + C(sex) + help_received'
model = smf.glm(
    formula=formula,
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=np.log(analysis_df['session_seconds'])
)
result = model.fit(cov_type='HC0')

# Overdispersion check
pearson_chi2 = result.pearson_chi2
resid_df = result.df_resid
overdispersion = pearson_chi2 / resid_df if resid_df > 0 else np.nan

# Group summaries
sex_group = analysis_df.groupby('sex')['rate_per_sec'].agg(['mean', 'median', 'count'])
help_group = analysis_df.groupby('help_received')['rate_per_sec'].agg(['mean', 'median', 'count'])

# Print key outputs
print('N:', len(analysis_df))
print('\nPoisson GLM (rate with offset) coefficients:')
print(result.summary())
print('\nRate ratios (exp(coef)) with robust SE:')
params = result.params
conf = result.conf_int()
rate_ratios = np.exp(params)
rr_conf = np.exp(conf)
rr_table = pd.DataFrame({
    'rate_ratio': rate_ratios,
    'ci_low': rr_conf[0],
    'ci_high': rr_conf[1],
    'p_value': result.pvalues
})
print(rr_table)

print('\nOverdispersion (Pearson chi2 / df):', overdispersion)

print('\nRate per second by sex:')
print(sex_group)

print('\nRate per second by help_received:')
print(help_group)

print('\nAge summary:')
print(analysis_df['age'].describe())

# Simple linear regression on log(rate+1e-6) for sensitivity
analysis_df['log_rate'] = np.log(analysis_df['rate_per_sec'] + 1e-6)
ols = smf.ols('log_rate ~ age + C(sex) + help_received', data=analysis_df).fit(cov_type='HC0')
print('\nOLS on log(rate) (robust SE):')
print(ols.summary())
