import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('hurricane.csv')

# Basic cleaning: ensure numeric columns
numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'ndam', 'ndam15', 'year']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Focus on rows with required fields
analysis_df = df[['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'year']].copy()
analysis_df = analysis_df.dropna()

# Outcome: log1p deaths
analysis_df['log_deaths'] = np.log1p(analysis_df['alldeaths'])

# Helper to fit OLS

def fit_ols(y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit()
    return model

# Model 1: log deaths ~ masfem
m1 = fit_ols(analysis_df['log_deaths'], analysis_df[['masfem']])

# Model 2: log deaths ~ masfem + intensity controls
m2 = fit_ols(
    analysis_df['log_deaths'],
    analysis_df[['masfem', 'wind', 'min', 'category', 'year']]
)

# Model 3: log deaths ~ gender_mf + controls
m3 = fit_ols(
    analysis_df['log_deaths'],
    analysis_df[['gender_mf', 'wind', 'min', 'category', 'year']]
)

# Compute correlations
corr_masfem_deaths = analysis_df['masfem'].corr(analysis_df['log_deaths'])

# Standardized effect for masfem in model 2 (beta = b * sd_x / sd_y)
masfem_sd = analysis_df['masfem'].std()
log_deaths_sd = analysis_df['log_deaths'].std()
masfem_beta = m2.params['masfem'] * masfem_sd / log_deaths_sd

# Print summary stats
print('N:', len(analysis_df))
print('Correlation masfem vs log deaths:', corr_masfem_deaths)
print('\nModel 1 (masfem only)\n', m1.summary())
print('\nModel 2 (masfem + controls)\n', m2.summary())
print('\nModel 3 (gender_mf + controls)\n', m3.summary())
print('\nStandardized beta for masfem (model 2):', masfem_beta)
