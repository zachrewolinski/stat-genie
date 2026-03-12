import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('hurricane.csv')

# Map columns to semantic meaning based on metadata + observed ranges
# Observed mapping:
# wind -> year of hurricane (1950-2012)
# alldeaths -> hurricane name (string)
# category -> femininity rating (1-11)
# ndam15 -> minimum pressure at landfall
# masfem_mturk -> binary gender (0 male, 1 female)
# gender_mf -> Saffir-Simpson category (1-5)
# name -> total deaths
# elapsedyrs -> normalized damage (2013)
# masfem -> years elapsed since hurricane
# min -> source
# ind -> MTurk femininity rating (1-11)
# year -> max wind speed
# source -> normalized damage (2015)

# Define variables

df = df.copy()

# Outcome: fatalities
outcome = df['name']

# Key predictors
fem_rating = df['category']
fem_rating_mturk = df['ind']
fem_binary = df['masfem_mturk']

# Controls (storm intensity + time)
controls = pd.DataFrame({
    'ssf_cat': df['gender_mf'],      # Saffir-Simpson category
    'max_wind': df['year'],          # max wind speed
    'min_pressure': df['ndam15'],    # min pressure at landfall
    'year': df['wind']               # year of hurricane
})

# Build dataset for modeling
model_df = pd.concat([
    outcome.rename('deaths'),
    fem_rating.rename('fem'),
    fem_rating_mturk.rename('fem_mturk'),
    fem_binary.rename('fem_bin'),
    controls
], axis=1)

# Drop missing values for each model separately

def fit_ols(y, X):
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit()
    return model

results = {}

# Use log1p to reduce skew
model_df['log_deaths'] = np.log1p(model_df['deaths'])

# Model A: bivariate (fem rating)
subset = model_df[['log_deaths', 'fem']].dropna()
results['A_fem_bivariate'] = fit_ols(subset['log_deaths'], subset[['fem']])

# Model B: controls (fem rating)
subset = model_df[['log_deaths', 'fem', 'ssf_cat', 'max_wind', 'min_pressure', 'year']].dropna()
results['B_fem_controls'] = fit_ols(subset['log_deaths'], subset[['fem', 'ssf_cat', 'max_wind', 'min_pressure', 'year']])

# Model C: bivariate (mturk fem rating)
subset = model_df[['log_deaths', 'fem_mturk']].dropna()
results['C_mturk_bivariate'] = fit_ols(subset['log_deaths'], subset[['fem_mturk']])

# Model D: controls (mturk fem rating)
subset = model_df[['log_deaths', 'fem_mturk', 'ssf_cat', 'max_wind', 'min_pressure', 'year']].dropna()
results['D_mturk_controls'] = fit_ols(subset['log_deaths'], subset[['fem_mturk', 'ssf_cat', 'max_wind', 'min_pressure', 'year']])

# Model E: binary gender bivariate
subset = model_df[['log_deaths', 'fem_bin']].dropna()
results['E_bin_bivariate'] = fit_ols(subset['log_deaths'], subset[['fem_bin']])

# Model F: binary gender with controls
subset = model_df[['log_deaths', 'fem_bin', 'ssf_cat', 'max_wind', 'min_pressure', 'year']].dropna()
results['F_bin_controls'] = fit_ols(subset['log_deaths'], subset[['fem_bin', 'ssf_cat', 'max_wind', 'min_pressure', 'year']])

# Summarize key coefficients and p-values
summary_rows = []
for key, model in results.items():
    if 'fem_mturk' in model.params.index:
        coef = model.params['fem_mturk']
        pval = model.pvalues['fem_mturk']
        var = 'fem_mturk'
    elif 'fem_bin' in model.params.index:
        coef = model.params['fem_bin']
        pval = model.pvalues['fem_bin']
        var = 'fem_bin'
    else:
        coef = model.params['fem']
        pval = model.pvalues['fem']
        var = 'fem'
    summary_rows.append({
        'model': key,
        'predictor': var,
        'coef': coef,
        'pval': pval,
        'n': int(model.nobs),
        'r2': model.rsquared
    })

summary = pd.DataFrame(summary_rows)

# Also compute simple correlation between fem rating and deaths
corr = model_df[['deaths', 'fem', 'fem_mturk', 'fem_bin']].corr()

# Save results to a JSON-ish text for manual inspection
summary.to_csv('model_summary.csv', index=False)

with open('model_summary.txt', 'w') as f:
    f.write('Model summary (coef on femininity predictors)\n')
    f.write(summary.to_string(index=False))
    f.write('\n\nCorrelation matrix (deaths vs fem measures)\n')
    f.write(corr.to_string())

print(summary)
print(corr)
