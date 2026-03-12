import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('panda_nuts.csv')

# Map columns based on metadata mismatch
# Actual variables inferred from descriptions and data types
# age_years -> hammer column
# sex -> nuts_opened column (m/f)
# help_received -> seconds column (y/N)
# nuts_opened_count -> help column
# session_seconds -> chimpanzee column

df = pd.DataFrame({
    'age_years': raw['hammer'],
    'sex': raw['nuts_opened'],
    'help_received': raw['seconds'],
    'nuts_opened': raw['help'],
    'seconds': raw['chimpanzee'],
})

# Clean help flag
help_map = {'y': 1, 'Y': 1, 'N': 0, 'n': 0}
df['help_received'] = df['help_received'].map(help_map)

# Drop any rows with missing mapping (should be none)
df = df.dropna()

# Rate (efficiency)
df['rate'] = df['nuts_opened'] / df['seconds']

# Basic summaries
print('Row count:', len(df))
print('Help value counts:')
print(df['help_received'].value_counts())
print('\nSex value counts:')
print(df['sex'].value_counts())

print('\nRate summary:')
print(df['rate'].describe())

# Compare mean rate by help and sex
print('\nMean rate by help:')
print(df.groupby('help_received')['rate'].mean())
print('\nMean rate by sex:')
print(df.groupby('sex')['rate'].mean())

# Correlation with age (Pearson)
print('\nCorrelation age vs rate:')
print(df[['age_years', 'rate']].corr().iloc[0,1])

# Poisson GLM with offset log(seconds)
# Use categorical for sex and help
# Add small constant to seconds to avoid log(0) (none should be zero)

df['log_seconds'] = np.log(df['seconds'])

formula = 'nuts_opened ~ age_years + C(sex) + help_received'

poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=df['log_seconds']
).fit()

print('\nPoisson GLM summary:')
print(poisson_model.summary())

# Check overdispersion: Pearson chi2 / df
pearson_chi2 = sum(poisson_model.resid_pearson**2)
pearson_ratio = pearson_chi2 / poisson_model.df_resid
print('\nOverdispersion ratio (Pearson chi2/df):', pearson_ratio)

# Negative Binomial GLM if overdispersed
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=df['log_seconds']
).fit()

print('\nNegative Binomial GLM summary:')
print(nb_model.summary())

# Also run OLS on log(rate+eps) as robustness

eps = 1e-6
df['log_rate'] = np.log(df['rate'] + eps)
ols_model = smf.ols('log_rate ~ age_years + C(sex) + help_received', data=df).fit()
print('\nOLS log(rate) summary:')
print(ols_model.summary())

# Save key results for later use

print('\nKey coefficients (NB):')
print(nb_model.params)
print('\nKey p-values (NB):')
print(nb_model.pvalues)
