import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Basic cleaning / transformations

df['log_deaths'] = np.log1p(df['alldeaths'])

# Main model: femininity rating with controls for storm intensity and damage
# Controls chosen to approximate storm strength/exposure
model1 = smf.ols('log_deaths ~ masfem + wind + min + ndam15 + year', data=df).fit(cov_type='HC3')

# Alternative model using binary gender indicator
model2 = smf.ols('log_deaths ~ gender_mf + wind + min + ndam15 + year', data=df).fit(cov_type='HC3')

# Simple correlation (no controls)
cor_masfem = df['masfem'].corr(df['log_deaths'])

print('N:', len(df))
print('Correlation masfem vs log_deaths:', cor_masfem)
print('\nModel 1 (masfem):')
print(model1.summary().tables[1])
print('\nModel 2 (gender_mf):')
print(model2.summary().tables[1])

# Save key results to a small dict for downstream use if needed
results = {
    'n': len(df),
    'cor_masfem_log_deaths': float(cor_masfem),
    'model1_masfem_coef': float(model1.params['masfem']),
    'model1_masfem_p': float(model1.pvalues['masfem']),
    'model2_gender_coef': float(model2.params['gender_mf']),
    'model2_gender_p': float(model2.pvalues['gender_mf']),
}

pd.Series(results).to_csv('analysis_results.csv')
