import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns to semantic meaning based on ranges/values
# year of hurricane
hurr_year = df['wind']
# hurricane name
hurr_name = df['alldeaths']
# femininity rating (1-11 scale)
masfem_index = df['category']
# alternative femininity rating (mturk)
masfem_mturk_cont = df['ind']
# binary female indicator
female_binary = df['masfem_mturk']
# hurricane category (1-5)
hurr_category = df['gender_mf']
# deaths
deaths = df['name']
# minimum pressure at landfall
min_pressure = df['ndam15']
# max wind speed at landfall
max_wind = df['year']

# Build analysis dataframe
analysis_df = pd.DataFrame({
    'deaths': deaths,
    'log_deaths': np.log1p(deaths),
    'masfem_index': masfem_index,
    'masfem_mturk_cont': masfem_mturk_cont,
    'female_binary': female_binary,
    'hurr_category': hurr_category,
    'min_pressure': min_pressure,
    'max_wind': max_wind,
    'year': hurr_year,
})

# Simple correlations
corrs = analysis_df[['deaths','log_deaths','masfem_index','masfem_mturk_cont','female_binary']].corr()
print('Correlations:')
print(corrs)

# Helper to run OLS with robust SE

def run_ols(y, X, label):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HC3')
    print(f"\nModel: {label}")
    print(model.summary().as_text())
    return model

# Model 1: log deaths ~ femininity (index)
run_ols(analysis_df['log_deaths'], analysis_df[['masfem_index']], 'log_deaths ~ masfem_index')

# Model 2: log deaths ~ femininity + controls
controls = analysis_df[['masfem_index','max_wind','min_pressure','hurr_category','year']]
run_ols(analysis_df['log_deaths'], controls, 'log_deaths ~ masfem_index + max_wind + min_pressure + category + year')

# Model 3: log deaths ~ mturk femininity + controls
controls2 = analysis_df[['masfem_mturk_cont','max_wind','min_pressure','hurr_category','year']]
run_ols(analysis_df['log_deaths'], controls2, 'log_deaths ~ masfem_mturk_cont + max_wind + min_pressure + category + year')

# Model 4: log deaths ~ female binary + controls
controls3 = analysis_df[['female_binary','max_wind','min_pressure','hurr_category','year']]
run_ols(analysis_df['log_deaths'], controls3, 'log_deaths ~ female_binary + max_wind + min_pressure + category + year')

# Also examine raw deaths with Poisson? Use GLM Poisson with robust SE
print('\nPoisson GLM (deaths) with controls and masfem_index')
X = sm.add_constant(analysis_df[['masfem_index','max_wind','min_pressure','hurr_category','year']])
poisson = sm.GLM(analysis_df['deaths'], X, family=sm.families.Poisson()).fit(cov_type='HC3')
print(poisson.summary().as_text())
