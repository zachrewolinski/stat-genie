import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Only relevant columns
cols = ['age','sex','help','nuts_opened','seconds']
df = df[cols].copy()

# Clean: drop missing or zero seconds
# Ensure numeric
for c in ['age','nuts_opened','seconds']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Standardize categories
# sex: keep as category
# help: map y/yes to 1, else 0

def normalize_help(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ['y','yes','1','true','t']:
        return 'y'
    if s in ['n','no','0','false','f']:
        return 'n'
    return s


df['help_norm'] = df['help'].apply(normalize_help)

# Efficiency: nuts per second
# avoid division by zero or negative

df = df[df['seconds'] > 0].copy()

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Drop missing

df_model = df.dropna(subset=['age','sex','help_norm','efficiency']).copy()

# Convert categories

df_model['sex'] = df_model['sex'].astype('category')
# help as category

df_model['help_norm'] = df_model['help_norm'].astype('category')

# OLS
model = smf.ols('efficiency ~ age + C(sex) + C(help_norm)', data=df_model).fit(cov_type='HC3')

# Also check log efficiency (add small constant if zeros)
min_eff = df_model['efficiency'].min()
add = 1e-6 if min_eff <= 0 else 0

if min_eff <= 0:
    df_model['log_eff'] = np.log(df_model['efficiency'] + add)
else:
    df_model['log_eff'] = np.log(df_model['efficiency'])

model_log = smf.ols('log_eff ~ age + C(sex) + C(help_norm)', data=df_model).fit(cov_type='HC3')

# Output summaries
print('N rows total:', len(df))
print('N rows model:', len(df_model))
print('Efficiency summary:')
print(df_model['efficiency'].describe())
print('\nOLS (efficiency) params and p-values:')
print(pd.DataFrame({'coef': model.params, 'p': model.pvalues, 'std_err': model.bse}))
print('\nOLS (log efficiency) params and p-values:')
print(pd.DataFrame({'coef': model_log.params, 'p': model_log.pvalues, 'std_err': model_log.bse}))

# Simple group means for help and sex
print('\nGroup means:')
print(df_model.groupby('help_norm')['efficiency'].mean())
print(df_model.groupby('sex')['efficiency'].mean())

# Correlation with age
print('\nAge correlation with efficiency:')
print(df_model[['age','efficiency']].corr().iloc[0,1])
