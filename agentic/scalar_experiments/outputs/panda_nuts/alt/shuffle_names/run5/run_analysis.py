import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('panda_nuts.csv')

# Map columns to actual meaning based on data distributions
# age column appears to be chimpanzee ID (22 unique)
# hammer column appears to be age in years (3-16)
# nuts_opened column appears to be sex (m/f)
# sex column appears to be hammer type (4 categories)
# help column appears to be number of nuts opened (count)
# chimpanzee column appears to be session duration in seconds
# seconds column appears to be help (y/N)

df = raw.rename(columns={
    'age': 'chimp_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'seconds',
    'seconds': 'help'
}).copy()

# Basic cleaning/encoding
# Ensure help is binary
# Sex as category, help as category

df['help'] = df['help'].map({'y': 1, 'N': 0})
# Some datasets may use uppercase/lowercase variants
if df['help'].isna().any():
    df['help'] = df['help'].fillna(df['seconds'].map({'Y': 1, 'y': 1, 'N': 0, 'n': 0}))

# Efficiency as nuts opened per second
# Guard against zero seconds (not expected)
df = df[df['seconds'] > 0].copy()
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Encode sex (m=1, f=0)
# Use category for regression with statsmodels formula

# Summary stats
print('Rows:', len(df))
print('Seconds min/max:', df['seconds'].min(), df['seconds'].max())
print('Nuts opened min/max:', df['nuts_opened'].min(), df['nuts_opened'].max())
print('\nGroup means (efficiency):')
print(df.groupby('sex')['efficiency'].mean())
print('\nGroup means (efficiency) by help:')
print(df.groupby('help')['efficiency'].mean())

# OLS on efficiency
ols = smf.ols('efficiency ~ age_years + C(sex) + help', data=df).fit()
print('\nOLS efficiency ~ age + sex + help')
print(ols.summary())

# Poisson GLM with offset log(seconds): nuts_opened counts
# Using exposure via offset
# Add small epsilon to seconds to avoid log(0)

df['log_seconds'] = np.log(df['seconds'])
poisson = smf.glm('nuts_opened ~ age_years + C(sex) + help', data=df,
                  family=sm.families.Poisson(), offset=df['log_seconds']).fit()
print('\nPoisson GLM nuts_opened with offset log(seconds)')
print(poisson.summary())

# Overdispersion check
pearson_chi2 = sum(poisson.resid_pearson**2)
ratio = pearson_chi2 / poisson.df_resid
print('\nOverdispersion ratio (Pearson chi2/df):', ratio)

# Negative Binomial GLM if overdispersed
try:
    nb = smf.glm('nuts_opened ~ age_years + C(sex) + help', data=df,
                 family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit()
    print('\nNegative Binomial GLM (alpha=1.0)')
    print(nb.summary())
except Exception as e:
    print('NB GLM failed:', e)

# Also check model with chimp_id fixed effects to account for repeated measures
# Use OLS on efficiency with chimp_id fixed effects
ols_fe = smf.ols('efficiency ~ age_years + C(sex) + help + C(chimp_id)', data=df).fit()
print('\nOLS with chimp_id fixed effects')
print(ols_fe.summary())

# Save key p-values for later
results = {
    'ols_pvalues': ols.pvalues.to_dict(),
    'poisson_pvalues': poisson.pvalues.to_dict(),
    'nb_pvalues': nb.pvalues.to_dict() if 'nb' in locals() else {},
    'ols_fe_pvalues': ols_fe.pvalues.to_dict(),
}
print('\nPvalues summary:', results)
