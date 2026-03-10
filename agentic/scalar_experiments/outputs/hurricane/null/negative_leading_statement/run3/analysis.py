import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Load data
csv_path = Path('hurricane.csv')
df = pd.read_csv(csv_path)

# Clean/prepare
# Use alldeaths; add 1 then log to reduce skew (common in literature)
df['log_deaths'] = np.log1p(df['alldeaths'])

# Basic correlation
corr = df['masfem'].corr(df['alldeaths'])

# Regression models
# Model 1: log deaths ~ masfem
X1 = sm.add_constant(df[['masfem']])
model1 = sm.OLS(df['log_deaths'], X1).fit(cov_type='HC3')

# Model 2: controls for storm severity (wind, min pressure, category)
# Note: lower pressure indicates stronger storm; include both wind and min.
X2 = sm.add_constant(df[['masfem', 'wind', 'min', 'category']])
model2 = sm.OLS(df['log_deaths'], X2).fit(cov_type='HC3')

# Model 3: add normalized damage (ndam15) as an additional proxy for severity/impact
# (Logged to reduce skew)
df['log_ndam15'] = np.log1p(df['ndam15'])
X3 = sm.add_constant(df[['masfem', 'wind', 'min', 'category', 'log_ndam15']])
model3 = sm.OLS(df['log_deaths'], X3).fit(cov_type='HC3')

results = {
    'n': int(df.shape[0]),
    'corr_masfem_alldeaths': float(corr),
    'model1': {
        'coef': float(model1.params['masfem']),
        'p': float(model1.pvalues['masfem']),
        'r2': float(model1.rsquared),
    },
    'model2': {
        'coef': float(model2.params['masfem']),
        'p': float(model2.pvalues['masfem']),
        'r2': float(model2.rsquared),
    },
    'model3': {
        'coef': float(model3.params['masfem']),
        'p': float(model3.pvalues['masfem']),
        'r2': float(model3.rsquared),
    },
}

print(json.dumps(results, indent=2))
