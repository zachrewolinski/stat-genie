import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('panda_nuts.csv')

# Map columns based on observed values and metadata mismatch
# age: raw['age']
# sex: raw['nuts_opened'] (f/m)
# help_received: raw['seconds'] (y/N)
# nuts_opened_count: raw['help']
# session_seconds: raw['chimpanzee']

df = pd.DataFrame({
    'age': raw['age'].astype(float),
    'sex': raw['nuts_opened'].astype(str),
    'help_received': raw['seconds'].astype(str),
    'nuts_opened': raw['help'].astype(float),
    'session_seconds': raw['chimpanzee'].astype(float),
})

# Clean help_received to binary 0/1 (y vs N)
df['help_received'] = df['help_received'].str.strip().str.lower().map({'y': 1, 'n': 0})

# Drop any rows with missing
clean = df.dropna().copy()

# Efficiency as nuts opened per second
clean['efficiency'] = clean['nuts_opened'] / clean['session_seconds']

# OLS on efficiency
ols = smf.ols('efficiency ~ age + C(sex) + help_received', data=clean).fit()

# Poisson regression on nuts opened with offset for time
# Add small constant to session_seconds to avoid log(0) though min is >0
clean['log_seconds'] = np.log(clean['session_seconds'])
poisson = smf.glm('nuts_opened ~ age + C(sex) + help_received', data=clean,
                  family=sm.families.Poisson(), offset=clean['log_seconds']).fit()

# Overdispersion check for Poisson; if overdispersed, use quasi (scale) for robust SE
# Use Pearson chi2 / df_resid
pearson_chi2 = sum(poisson.resid_pearson**2)
ratio = pearson_chi2 / poisson.df_resid

# Negative binomial as robustness
try:
    nb = smf.glm('nuts_opened ~ age + C(sex) + help_received', data=clean,
                 family=sm.families.NegativeBinomial(alpha=1.0), offset=clean['log_seconds']).fit()
except Exception:
    nb = None

print('Rows:', len(clean))
print('\nOLS efficiency')
print(ols.summary())
print('\nPoisson with offset')
print(poisson.summary())
print('\nPoisson overdispersion ratio (Pearson chi2/df):', ratio)
if nb is not None:
    print('\nNegative binomial with offset')
    print(nb.summary())

# Also compute group means for interpretability
print('\nMean efficiency by sex and help')
print(clean.groupby(['sex','help_received'])['efficiency'].mean())

