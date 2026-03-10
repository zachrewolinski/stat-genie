import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('hurricane.csv')

# Rename columns based on info.json descriptions
rename_map = {
    'ndam': 'id',
    'wind': 'year',
    'alldeaths': 'hurricane_name',
    'category': 'femininity_index',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'saffir_simpson_cat',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'min': 'data_source',
    'ind': 'femininity_mturk',
    'year': 'max_wind_speed',
    'source': 'damage_2015',
}

df = df.rename(columns=rename_map)

# Basic cleaning: ensure numeric columns are numeric
num_cols = [
    'year', 'femininity_index', 'min_pressure', 'female_binary', 'saffir_simpson_cat',
    'deaths', 'damage_2013', 'years_elapsed', 'femininity_mturk',
    'max_wind_speed', 'damage_2015'
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Define outcome and predictors
# Log-transform deaths to reduce skew
# Add 1 to handle zeros

df['log_deaths'] = np.log1p(df['deaths'])

# Correlations (Spearman) between femininity and deaths
spearman_fem = stats.spearmanr(df['femininity_index'], df['deaths'], nan_policy='omit')
spearman_fem_log = stats.spearmanr(df['femininity_index'], df['log_deaths'], nan_policy='omit')

spearman_bin = stats.spearmanr(df['female_binary'], df['deaths'], nan_policy='omit')
spearman_bin_log = stats.spearmanr(df['female_binary'], df['log_deaths'], nan_policy='omit')

# Regression with controls for storm severity and year
# Controls: max_wind_speed, min_pressure, saffir_simpson_cat, year
# We include year to adjust for temporal trends in reporting/preparedness.
controls = ['max_wind_speed', 'min_pressure', 'saffir_simpson_cat', 'year']

# Model 1: femininity index
X1 = df[['femininity_index'] + controls]
X1 = sm.add_constant(X1)
model1 = sm.OLS(df['log_deaths'], X1, missing='drop').fit()

# Model 2: female binary
X2 = df[['female_binary'] + controls]
X2 = sm.add_constant(X2)
model2 = sm.OLS(df['log_deaths'], X2, missing='drop').fit()

# Alternative count model: Poisson with log link
# If convergence issues, catch and proceed
poisson_results = {}
for label, X in [('femininity_index', X1), ('female_binary', X2)]:
    try:
        poisson_model = sm.GLM(df['deaths'], X, family=sm.families.Poisson(), missing='drop').fit()
        poisson_results[label] = poisson_model
    except Exception as e:
        poisson_results[label] = e

# Print key results
print('N', len(df))
print('Deaths summary', df['deaths'].describe())
print('Spearman femininity_index vs deaths:', spearman_fem)
print('Spearman femininity_index vs log_deaths:', spearman_fem_log)
print('Spearman female_binary vs deaths:', spearman_bin)
print('Spearman female_binary vs log_deaths:', spearman_bin_log)

print('\nOLS log_deaths ~ femininity_index + controls')
print(model1.summary().tables[1])
print('\nOLS log_deaths ~ female_binary + controls')
print(model2.summary().tables[1])

if isinstance(poisson_results.get('femininity_index'), Exception):
    print('\nPoisson femininity_index model error:', poisson_results['femininity_index'])
else:
    print('\nPoisson deaths ~ femininity_index + controls')
    print(poisson_results['femininity_index'].summary().tables[1])

if isinstance(poisson_results.get('female_binary'), Exception):
    print('\nPoisson female_binary model error:', poisson_results['female_binary'])
else:
    print('\nPoisson deaths ~ female_binary + controls')
    print(poisson_results['female_binary'].summary().tables[1])
