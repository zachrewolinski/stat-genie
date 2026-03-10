import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Prepare variables
# Use log1p of deaths to handle skew and zeros

df['log_deaths'] = np.log1p(df['alldeaths'])

# Select relevant columns and drop missing values for model
cols = ['alldeaths', 'log_deaths', 'masfem', 'wind', 'min', 'category', 'year', 'gender_mf', 'masfem_mturk']

model_df = df[cols].dropna()

# Function to fit OLS with robust SEs

def fit_ols(y, X):
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model

results = {}

# Model 1: log_deaths ~ masfem
m1 = fit_ols(model_df['log_deaths'], model_df[['masfem']])
results['m1'] = m1

# Model 2: controls for severity and year
m2 = fit_ols(model_df['log_deaths'], model_df[['masfem', 'wind', 'min', 'category', 'year']])
results['m2'] = m2

# Model 3: using gender_mf instead of masfem
m3 = fit_ols(model_df['log_deaths'], model_df[['gender_mf', 'wind', 'min', 'category', 'year']])
results['m3'] = m3

# Model 4: use alternative femininity rating
m4 = fit_ols(model_df['log_deaths'], model_df[['masfem_mturk', 'wind', 'min', 'category', 'year']])
results['m4'] = m4

# Extract key stats
summary = {}
for key, model in results.items():
    coef = model.params
    pvals = model.pvalues
    summary[key] = {
        'n': int(model.nobs),
        'r2': float(model.rsquared),
        'coef': coef.to_dict(),
        'pval': pvals.to_dict(),
    }

# Correlations
corr = model_df[['masfem', 'gender_mf', 'masfem_mturk', 'alldeaths', 'log_deaths']].corr()

print('SUMMARY')
for k, v in summary.items():
    print('\n', k)
    print('n', v['n'], 'r2', round(v['r2'], 4))
    for var in ['masfem', 'gender_mf', 'masfem_mturk']:
        if var in v['coef']:
            print(var, 'coef', round(v['coef'][var], 4), 'p', round(v['pval'][var], 4))

print('\nCORR')
print(corr)
