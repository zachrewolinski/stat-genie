import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Relevant columns
cols = [
    'alldeaths',
    'masfem',
    'gender_mf',
    'wind',
    'min',
    'category',
    'ndam15',
    'year'
]

df = _df[cols].copy()

# Ensure numeric
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing values
model_df = df.dropna().copy()

# Log deaths for OLS
model_df['log_deaths'] = np.log1p(model_df['alldeaths'])

# OLS with robust SE
formula_masfem = 'log_deaths ~ masfem + wind + min + category + ndam15 + year'
model_masfem = smf.ols(formula_masfem, data=model_df).fit(cov_type='HC3')

formula_gender = 'log_deaths ~ gender_mf + wind + min + category + ndam15 + year'
model_gender = smf.ols(formula_gender, data=model_df).fit(cov_type='HC3')

# Poisson GLM (count outcome)
poisson_formula = 'alldeaths ~ masfem + wind + min + category + ndam15 + year'
poisson_model = smf.glm(poisson_formula, data=model_df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Overdispersion check
mean_deaths = model_df['alldeaths'].mean()
var_deaths = model_df['alldeaths'].var(ddof=1)
alpha_hat = max((var_deaths - mean_deaths) / (mean_deaths ** 2), 1e-8)

# Negative Binomial GLM using moment-based alpha
nb_mom = smf.glm(
    poisson_formula,
    data=model_df,
    family=sm.families.NegativeBinomial(alpha=alpha_hat)
).fit(cov_type='HC3')

print('N:', len(model_df))
print(f"Mean deaths: {mean_deaths:.2f}, Variance deaths: {var_deaths:.2f}")
print(f"Moment-based alpha (overdispersion): {alpha_hat:.4f}")

print('\nOLS log1p deaths ~ masfem + controls')
print(model_masfem.summary().tables[1])

print('\nOLS log1p deaths ~ gender_mf + controls')
print(model_gender.summary().tables[1])

print('\nPoisson alldeaths ~ masfem + controls')
print(poisson_model.summary().tables[1])

print('\nNegBin (GLM, moment alpha) alldeaths ~ masfem + controls')
print(nb_mom.summary().tables[1])
