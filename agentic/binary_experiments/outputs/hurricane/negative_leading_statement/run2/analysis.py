import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'hurricane.csv'
df = pd.read_csv(DF_PATH)

# Prepare variables
# Use log(1 + deaths) to reduce skew in fatalities

df['log_deaths'] = np.log1p(df['alldeaths'])

# Simple correlations
corr_pearson = df['masfem'].corr(df['alldeaths'])
corr_spearman = df['masfem'].corr(df['alldeaths'], method='spearman')

# OLS: log deaths on femininity + controls
ols_cols = ['masfem', 'wind', 'min', 'category', 'year', 'ndam15']
X = sm.add_constant(df[ols_cols])
ols = sm.OLS(df['log_deaths'], X, missing='drop').fit(cov_type='HC3')

# Negative Binomial: deaths on femininity + controls (handles overdispersion)
nb = smf.glm(
    'alldeaths ~ masfem + wind + min + category + year + ndam15',
    data=df,
    family=sm.families.NegativeBinomial()
).fit()

# Poisson (for comparison; sensitive to overdispersion)
pois = smf.glm(
    'alldeaths ~ masfem + wind + min + category + year + ndam15',
    data=df,
    family=sm.families.Poisson()
).fit()

# Print a concise summary for interpretation
print('Correlation (Pearson) masfem vs deaths:', corr_pearson)
print('Correlation (Spearman) masfem vs deaths:', corr_spearman)

print('\nOLS log(deaths) results (robust SE):')
print(ols.summary().tables[1])

print('\nNegative Binomial results:')
print(nb.summary().tables[1])

print('\nPoisson results (note: can be optimistic if overdispersed):')
print(pois.summary().tables[1])
