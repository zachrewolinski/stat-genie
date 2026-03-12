import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns
# feature4: masculinity-femininity index (higher = more feminine)
# feature6: binary gender indicator (0 male, 1 female)
# feature8: deaths
# feature7: category
# feature5: min pressure
# feature13: max wind speed

# Basic clean
cols = ['feature4','feature6','feature8','feature7','feature5','feature13','feature9','feature14']

# ensure numeric
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Create log deaths
(df['feature8']>=0).all()
df['log_deaths'] = np.log1p(df['feature8'])

# simple correlations
corr_fem_deaths = df['feature4'].corr(df['feature8'])
corr_fem_log = df['feature4'].corr(df['log_deaths'])

# regressions
# model 1: log deaths ~ femininity
X1 = sm.add_constant(df[['feature4']])
model1 = sm.OLS(df['log_deaths'], X1).fit()

# model 2: log deaths ~ femininity + category + wind + pressure
X2 = sm.add_constant(df[['feature4','feature7','feature13','feature5']])
model2 = sm.OLS(df['log_deaths'], X2).fit()

# model 3: log deaths ~ binary female + controls
X3 = sm.add_constant(df[['feature6','feature7','feature13','feature5']])
model3 = sm.OLS(df['log_deaths'], X3).fit()

# model 4: deaths (count) maybe overdispersed; use negative binomial? but keep OLS for quick.

summary = {
    'n': int(df.shape[0]),
    'corr_fem_deaths': float(corr_fem_deaths),
    'corr_fem_log': float(corr_fem_log),
    'model1': {
        'coef_fem': float(model1.params['feature4']),
        'p_fem': float(model1.pvalues['feature4']),
        'r2': float(model1.rsquared)
    },
    'model2': {
        'coef_fem': float(model2.params['feature4']),
        'p_fem': float(model2.pvalues['feature4']),
        'r2': float(model2.rsquared)
    },
    'model3': {
        'coef_female': float(model3.params['feature6']),
        'p_female': float(model3.pvalues['feature6']),
        'r2': float(model3.rsquared)
    }
}

print(json.dumps(summary, indent=2))
