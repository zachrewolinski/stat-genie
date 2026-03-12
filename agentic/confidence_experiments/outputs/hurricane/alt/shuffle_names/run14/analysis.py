import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns based on ranges and known dataset
# femininity rating: category
fem = _df['category']
# binary female indicator
female = _df['masfem_mturk']
# deaths
deaths = _df['name']
# year
year = _df['wind']
# wind speed
wind = _df['year']
# min pressure
pressure = _df['ndam15']
# damage (normalized)
damage = _df['source']
# category (Saffir-Simpson)
sscat = _df['gender_mf']

# Basic stats
print('N', len(_df))
print('Deaths summary', deaths.describe())
print('Fem summary', fem.describe())

# Correlation (Spearman) between femininity and deaths
spearman = stats.spearmanr(fem, deaths)
print('Spearman rho', spearman)

# OLS on log deaths
_df['log_deaths'] = np.log1p(deaths)
X = pd.DataFrame({
    'fem': fem,
    'wind': wind,
    'pressure': pressure,
    'sscat': sscat,
    'damage': np.log1p(damage),
    'year': year,
})
X = sm.add_constant(X)
model = sm.OLS(_df['log_deaths'], X).fit()
print(model.summary())

# Alternative: using female indicator
X2 = pd.DataFrame({
    'female': female,
    'wind': wind,
    'pressure': pressure,
    'sscat': sscat,
    'damage': np.log1p(damage),
    'year': year,
})
X2 = sm.add_constant(X2)
model2 = sm.OLS(_df['log_deaths'], X2).fit()
print(model2.summary())

# Simple bivariate
model3 = sm.OLS(_df['log_deaths'], sm.add_constant(fem)).fit()
print(model3.summary())
