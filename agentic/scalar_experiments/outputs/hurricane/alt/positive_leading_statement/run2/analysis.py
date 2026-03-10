import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Prepare variables
# Log-transform deaths and damages to reduce skew
for col in ['alldeaths', 'ndam', 'ndam15']:
    df[f'log_{col}'] = np.log1p(df[col])

# Some models use masfem (continuous femininity rating)
# Controls for severity: wind, min pressure, category, damages, year

models = {}

# Base model with masfem and key controls
formula1 = 'log_alldeaths ~ masfem + wind + min + category + log_ndam15 + year'
models['masfem_controls'] = smf.ols(formula1, data=df).fit()

# Alternative using gender_mf (binary)
formula2 = 'log_alldeaths ~ gender_mf + wind + min + category + log_ndam15 + year'
models['gender_controls'] = smf.ols(formula2, data=df).fit()

# Using MTurk femininity ratings
formula3 = 'log_alldeaths ~ masfem_mturk + wind + min + category + log_ndam15 + year'
models['masfem_mturk_controls'] = smf.ols(formula3, data=df).fit()

# A simpler model without damages (avoid post-treatment concerns)
formula4 = 'log_alldeaths ~ masfem + wind + min + category + year'
models['masfem_no_damage'] = smf.ols(formula4, data=df).fit()

# Correlation between masfem and deaths
corr = df[['masfem', 'alldeaths']].corr().iloc[0,1]

# Print key results
print('N:', len(df))
print('Correlation masfem vs deaths:', corr)

for name, model in models.items():
    coef = model.params.get('masfem', model.params.get('gender_mf', model.params.get('masfem_mturk', np.nan)))
    pval = model.pvalues.get('masfem', model.pvalues.get('gender_mf', model.pvalues.get('masfem_mturk', np.nan)))
    print(f'\nModel: {name}')
    print('coef:', coef)
    print('p-value:', pval)
    print('R2:', model.rsquared)
    # For context, also print std err and CI for the focal variable
    if 'masfem' in model.params:
        se = model.bse['masfem']
        ci = model.conf_int().loc['masfem'].tolist()
        print('SE:', se, 'CI:', ci)
    elif 'gender_mf' in model.params:
        se = model.bse['gender_mf']
        ci = model.conf_int().loc['gender_mf'].tolist()
        print('SE:', se, 'CI:', ci)
    elif 'masfem_mturk' in model.params:
        se = model.bse['masfem_mturk']
        ci = model.conf_int().loc['masfem_mturk'].tolist()
        print('SE:', se, 'CI:', ci)

