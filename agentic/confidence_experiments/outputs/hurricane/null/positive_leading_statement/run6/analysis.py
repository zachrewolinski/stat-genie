import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic preprocessing
# log1p of deaths to handle skew and zeros
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Femininity measure: masfem (1-11, higher = more feminine)

# Bivariate association
corr, corr_p = stats.pearsonr(_df['masfem'], _df['log_deaths'])

# Regression models
# Model 1: log_deaths ~ masfem
X1 = sm.add_constant(_df[['masfem']])
model1 = sm.OLS(_df['log_deaths'], X1).fit(cov_type='HC3')

# Model 2: add storm intensity controls (wind, min pressure, category)
X2 = sm.add_constant(_df[['masfem', 'wind', 'min', 'category']])
model2 = sm.OLS(_df['log_deaths'], X2).fit(cov_type='HC3')

# Model 3: add damage as a proxy for exposure/impact (ndam15)
# log transform damage to reduce skew
_df['log_ndam15'] = np.log1p(_df['ndam15'])
X3 = sm.add_constant(_df[['masfem', 'wind', 'min', 'category', 'log_ndam15']])
model3 = sm.OLS(_df['log_deaths'], X3).fit(cov_type='HC3')

# Also test binary gender indicator
X4 = sm.add_constant(_df[['gender_mf', 'wind', 'min', 'category']])
model4 = sm.OLS(_df['log_deaths'], X4).fit(cov_type='HC3')

# Save key results
results = {
    'n': int(_df.shape[0]),
    'corr_masfem_logdeaths': float(corr),
    'corr_p': float(corr_p),
    'model1': {
        'coef_masfem': float(model1.params['masfem']),
        'p_masfem': float(model1.pvalues['masfem']),
        'r2': float(model1.rsquared)
    },
    'model2': {
        'coef_masfem': float(model2.params['masfem']),
        'p_masfem': float(model2.pvalues['masfem']),
        'r2': float(model2.rsquared)
    },
    'model3': {
        'coef_masfem': float(model3.params['masfem']),
        'p_masfem': float(model3.pvalues['masfem']),
        'r2': float(model3.rsquared)
    },
    'model4_gender': {
        'coef_gender_mf': float(model4.params['gender_mf']),
        'p_gender_mf': float(model4.pvalues['gender_mf']),
        'r2': float(model4.rsquared)
    },
}

print(json.dumps(results, indent=2))
