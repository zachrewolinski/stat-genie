import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map shuffled column names to semantics based on info.json descriptions
# age -> chimpanzee ID
# hammer -> age in years
# nuts_opened -> sex
# sex -> hammer type
# help -> nuts opened count
# chimpanzee -> session duration in seconds
# seconds -> received help (yes/no)

df = df.rename(columns={
    'age': 'chimpanzee_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'duration_seconds',
    'seconds': 'helped'
})

# Clean / encode
# Ensure proper types

df['sex'] = df['sex'].astype(str).str.strip().str.lower()
# Standardize sex labels: assume 'f'/'m'

# help variable is y/N per metadata; normalize to 1/0

df['helped'] = df['helped'].astype(str).str.strip().str.lower()

def map_help(x):
    if x in ['y', 'yes', '1', 'true', 't']:
        return 1
    if x in ['n', 'no', '0', 'false', 'f']:
        return 0
    return np.nan

df['helped_bin'] = df['helped'].map(map_help)

# Compute efficiency: nuts opened per second
# Avoid divide by zero

df = df[df['duration_seconds'] > 0].copy()
df['efficiency'] = df['nuts_opened'] / df['duration_seconds']

# Summary stats
summary = df[['age_years','sex','helped_bin','nuts_opened','duration_seconds','efficiency']].describe(include='all')

# OLS on log efficiency (add small constant)
# Some efficiencies could be zero if nuts_opened=0; add small constant

df['log_efficiency'] = np.log(df['efficiency'] + 1e-6)

ols_model = smf.ols('log_efficiency ~ age_years + C(sex) + helped_bin', data=df).fit()

# Poisson regression with offset to model rate (nuts_opened / duration)
# Use nuts_opened count with log(duration) offset

# Check for zeros and overdisp; but still can fit Poisson
poisson_model = smf.glm(
    'nuts_opened ~ age_years + C(sex) + helped_bin',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['duration_seconds'])
).fit()

# Also fit Negative Binomial to check robustness
nb_model = smf.glm(
    'nuts_opened ~ age_years + C(sex) + helped_bin',
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['duration_seconds'])
).fit()

# Gather key results
results = {
    'n': len(df),
    'sex_levels': df['sex'].unique().tolist(),
    'helped_counts': df['helped_bin'].value_counts(dropna=False).to_dict(),
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_median': df['efficiency'].median(),
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_model.pvalues.to_dict(),
    'ols_r2': ols_model.rsquared,
    'poisson_params': poisson_model.params.to_dict(),
    'poisson_pvalues': poisson_model.pvalues.to_dict(),
    'nb_params': nb_model.params.to_dict(),
    'nb_pvalues': nb_model.pvalues.to_dict(),
}

print('SUMMARY')
print(summary)
print('\nOLS')
print(ols_model.summary())
print('\nPOISSON')
print(poisson_model.summary())
print('\nNEGATIVE_BINOMIAL')
print(nb_model.summary())
print('\nRESULTS_DICT')
print(results)
