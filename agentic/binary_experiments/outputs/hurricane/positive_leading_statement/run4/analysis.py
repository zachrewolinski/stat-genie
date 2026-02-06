import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Outcome: log deaths to reduce skew; add 1 to handle zeros
# Main predictor: masfem (higher = more feminine)
# Controls: wind, min pressure, category, damage (ndam15), year
cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'year']
model_df = df[cols].dropna().copy()
model_df['log_deaths'] = np.log1p(model_df['alldeaths'])

X = model_df[['masfem', 'wind', 'min', 'category', 'ndam15', 'year']]
X = sm.add_constant(X)
y = model_df['log_deaths']

ols = sm.OLS(y, X).fit(cov_type='HC3')

# Also fit a count model (Negative Binomial) as a robustness check
# Use deaths as count with same predictors
nb = sm.GLM(
    model_df['alldeaths'],
    X,
    family=sm.families.NegativeBinomial()
).fit()

# Simple bivariate correlation
corr = model_df[['alldeaths', 'masfem']].corr().iloc[0, 1]

print('OLS (log deaths) HC3 summary')
print(ols.summary())
print('\nNegative Binomial summary')
print(nb.summary())
print('\nBivariate correlation (alldeaths vs masfem):', corr)

# Extract key stats for conclusion convenience
coef = ols.params['masfem']
pval = ols.pvalues['masfem']
print('\nOLS masfem coef:', coef, 'p-value:', pval)

coef_nb = nb.params['masfem']
pval_nb = nb.pvalues['masfem']
print('NB masfem coef:', coef_nb, 'p-value:', pval_nb)
