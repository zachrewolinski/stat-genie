import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Select relevant columns
cols = ['masfem','gender_mf','alldeaths','wind','min','category','year','elapsedyrs','ndam','ndam15']
missing_cols = [c for c in cols if c not in _df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Prepare dataset
# Use log1p of deaths to handle skew and zeros
_df = _df.copy()
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Basic correlation
corr_masfem = _df['masfem'].corr(_df['log_deaths'])

# Model with masfem
X1 = _df[['masfem','wind','min','category','year']].copy()
X1 = sm.add_constant(X1)
model1 = sm.OLS(_df['log_deaths'], X1).fit(cov_type='HC3')

# Model with gender_mf (binary female)
X2 = _df[['gender_mf','wind','min','category','year']].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(_df['log_deaths'], X2).fit(cov_type='HC3')

# Model with masfem and gender indicator (collinearity risk but both included to check)
X3 = _df[['masfem','gender_mf','wind','min','category','year']].copy()
X3 = sm.add_constant(X3)
model3 = sm.OLS(_df['log_deaths'], X3).fit(cov_type='HC3')

# Also check negative binomial? not needed

# Print key stats
print('n=', len(_df))
print('corr masfem vs log_deaths:', corr_masfem)

for i, m in enumerate([model1, model2, model3], start=1):
    print('\nModel', i)
    print(m.summary().tables[1])

# Extract coefficients and p-values for masfem/gender_mf
res = {
    'model1_masfem_coef': model1.params.get('masfem'),
    'model1_masfem_p': model1.pvalues.get('masfem'),
    'model2_gender_coef': model2.params.get('gender_mf'),
    'model2_gender_p': model2.pvalues.get('gender_mf'),
    'model3_masfem_coef': model3.params.get('masfem'),
    'model3_masfem_p': model3.pvalues.get('masfem'),
    'model3_gender_coef': model3.params.get('gender_mf'),
    'model3_gender_p': model3.pvalues.get('gender_mf'),
}
print('\nKey results:', res)

# Save results for later use
import json
with open('analysis_results.json','w') as f:
    json.dump({
        'n': len(_df),
        'corr_masfem_logdeaths': corr_masfem,
        'model1': {'coef': res['model1_masfem_coef'], 'p': res['model1_masfem_p']},
        'model2': {'coef': res['model2_gender_coef'], 'p': res['model2_gender_p']},
        'model3': {
            'masfem_coef': res['model3_masfem_coef'], 'masfem_p': res['model3_masfem_p'],
            'gender_coef': res['model3_gender_coef'], 'gender_p': res['model3_gender_p']
        }
    }, f, indent=2)
