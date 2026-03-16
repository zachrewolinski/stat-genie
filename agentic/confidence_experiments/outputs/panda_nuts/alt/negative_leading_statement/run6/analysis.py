import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure consistent categories
# Efficiency: nuts opened per second
# Handle any zero seconds by setting efficiency to NaN

df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')

# Efficiency
# Avoid division by zero

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Clean categorical vars
# Standardize sex and help to lower-case

df['sex'] = df['sex'].astype(str).str.strip().str.lower()
df['help'] = df['help'].astype(str).str.strip().str.lower()

# Recode help: y/yes -> 1, n/no -> 0
# The dataset appears to use 'y' and 'n' or 'N'
help_map = {'y': 1, 'yes': 1, 'n': 0, 'no': 0}
df['help_bin'] = df['help'].map(help_map)

# Drop rows with missing key variables
analysis_df = df.dropna(subset=['efficiency', 'age', 'sex', 'help_bin'])

# Descriptives
print('Rows:', len(analysis_df))
print('Unique chimps:', analysis_df['chimpanzee'].nunique())
print('Efficiency summary:', analysis_df['efficiency'].describe())

# OLS with categorical sex and help
ols_model = smf.ols('efficiency ~ age + C(sex) + help_bin', data=analysis_df).fit(cov_type='HC3')
print('\nOLS (HC3 robust SE)')
print(ols_model.summary())

# Mixed effects model with chimpanzee random intercept, if possible
try:
    mixed_model = smf.mixedlm('efficiency ~ age + C(sex) + help_bin', data=analysis_df, groups=analysis_df['chimpanzee']).fit(reml=False, method='lbfgs')
    print('\nMixedLM (random intercept by chimpanzee)')
    print(mixed_model.summary())
except Exception as e:
    print('MixedLM failed:', e)

# Also check model with log efficiency (if positive)
analysis_df = analysis_df.copy()
analysis_df['efficiency_log'] = np.log(analysis_df['efficiency'].replace(0, np.nan))
log_df = analysis_df.dropna(subset=['efficiency_log'])
if len(log_df) > 0:
    log_model = smf.ols('efficiency_log ~ age + C(sex) + help_bin', data=log_df).fit(cov_type='HC3')
    print('\nOLS on log efficiency (HC3 robust SE)')
    print(log_model.summary())

