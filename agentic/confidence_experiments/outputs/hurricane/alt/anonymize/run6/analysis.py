import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning: ensure numeric for key columns
# Features mapping from info.json
# feature4: masfem index (1=very masculine, 11=very feminine)
# feature6: binary gender indicator (0 male, 1 female)
# feature8: deaths
# feature5: min pressure, feature13: max wind speed, feature7: category

for col in ['feature4','feature6','feature8','feature5','feature13','feature7','feature12']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key variables
key_cols = ['feature4','feature8','feature5','feature13','feature7','feature6']
df_key = df.dropna(subset=key_cols).copy()

# Outcome: log1p deaths to reduce skew

df_key['log_deaths'] = np.log1p(df_key['feature8'])

# Simple correlation
corr = df_key['feature4'].corr(df_key['feature8'])

# OLS models
# Model 1: log deaths ~ femininity index
model1 = smf.ols('log_deaths ~ feature4', data=df_key).fit()

# Model 2: add storm intensity controls
model2 = smf.ols('log_deaths ~ feature4 + feature13 + feature5 + feature7', data=df_key).fit()

# Model 3: using binary gender indicator
model3 = smf.ols('log_deaths ~ feature6 + feature13 + feature5 + feature7', data=df_key).fit()

# Alternative masfem rating from MTurk (feature12)
model4 = smf.ols('log_deaths ~ feature12 + feature13 + feature5 + feature7', data=df_key.dropna(subset=['feature12'])).fit()

results = {
    'n': int(df_key.shape[0]),
    'corr_feature4_deaths': corr,
    'model1': {
        'coef_feature4': model1.params.get('feature4', np.nan),
        'p_feature4': model1.pvalues.get('feature4', np.nan),
        'r2': model1.rsquared,
    },
    'model2': {
        'coef_feature4': model2.params.get('feature4', np.nan),
        'p_feature4': model2.pvalues.get('feature4', np.nan),
        'r2': model2.rsquared,
    },
    'model3': {
        'coef_feature6': model3.params.get('feature6', np.nan),
        'p_feature6': model3.pvalues.get('feature6', np.nan),
        'r2': model3.rsquared,
    },
    'model4': {
        'coef_feature12': model4.params.get('feature12', np.nan),
        'p_feature12': model4.pvalues.get('feature12', np.nan),
        'r2': model4.rsquared,
    },
}

print(json.dumps(results, indent=2, default=float))
