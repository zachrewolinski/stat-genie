import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data

df = pd.read_csv('hurricane.csv')

# Basic prep
# Use log1p for skewed outcomes
for col in ['alldeaths','ndam','ndam15']:
    df[f'log1p_{col}'] = np.log1p(df[col])

# Add post-1979 indicator
# Storm naming with male/female started 1979

post79 = df['year'] >= 1979

# Models to evaluate effect of femininity on deaths and damage controlling for intensity

models = {}

# Controls for intensity and time
controls = ['wind','min','category','year']

# Build formula strings

formula_deaths_masfem = 'log1p_alldeaths ~ masfem + ' + ' + '.join(controls)
formula_deaths_gender = 'log1p_alldeaths ~ gender_mf + ' + ' + '.join(controls)

formula_dam_masfem = 'log1p_ndam15 ~ masfem + ' + ' + '.join(controls)
formula_dam_gender = 'log1p_ndam15 ~ gender_mf + ' + ' + '.join(controls)

# Fit on full sample
models['deaths_masfem_full'] = ols(formula_deaths_masfem, data=df).fit(cov_type='HC3')
models['deaths_gender_full'] = ols(formula_deaths_gender, data=df).fit(cov_type='HC3')
models['dam_masfem_full'] = ols(formula_dam_masfem, data=df).fit(cov_type='HC3')
models['dam_gender_full'] = ols(formula_dam_gender, data=df).fit(cov_type='HC3')

# Fit on post-1979 subset to reduce confounding by era

df_post = df[post79].copy()
models['deaths_masfem_post79'] = ols(formula_deaths_masfem, data=df_post).fit(cov_type='HC3')
models['deaths_gender_post79'] = ols(formula_deaths_gender, data=df_post).fit(cov_type='HC3')
models['dam_masfem_post79'] = ols(formula_dam_masfem, data=df_post).fit(cov_type='HC3')
models['dam_gender_post79'] = ols(formula_dam_gender, data=df_post).fit(cov_type='HC3')

# Also simple correlations
corrs = {
    'masfem_alldeaths': df['masfem'].corr(df['alldeaths']),
    'gender_alldeaths': df['gender_mf'].corr(df['alldeaths']),
    'masfem_logdeaths': df['masfem'].corr(df['log1p_alldeaths']),
    'gender_logdeaths': df['gender_mf'].corr(df['log1p_alldeaths']),
    'masfem_ndam15': df['masfem'].corr(df['ndam15']),
    'gender_ndam15': df['gender_mf'].corr(df['ndam15']),
}

# Print key results
print('N full:', len(df), 'N post79:', len(df_post))
print('Correlations:')
for k, v in corrs.items():
    print(k, v)

print('\nRegression summaries (coef, pval) for name gender effects:')
for name, model in models.items():
    if 'masfem' in name:
        coef = model.params.get('masfem', np.nan)
        pval = model.pvalues.get('masfem', np.nan)
    else:
        coef = model.params.get('gender_mf', np.nan)
        pval = model.pvalues.get('gender_mf', np.nan)
    print(name, 'coef=', coef, 'p=', pval, 'R2=', model.rsquared)

# Save key metrics to a JSON-like text for downstream use if needed
import json
out = {
    'n_full': len(df),
    'n_post79': len(df_post),
    'correlations': corrs,
    'results': {}
}
for name, model in models.items():
    if 'masfem' in name:
        coef = float(model.params.get('masfem', np.nan))
        pval = float(model.pvalues.get('masfem', np.nan))
        se = float(model.bse.get('masfem', np.nan))
    else:
        coef = float(model.params.get('gender_mf', np.nan))
        pval = float(model.pvalues.get('gender_mf', np.nan))
        se = float(model.bse.get('gender_mf', np.nan))
    out['results'][name] = {
        'coef': coef,
        'se': se,
        'pval': pval,
        'rsq': float(model.rsquared)
    }

with open('analysis_results.json','w') as f:
    json.dump(out, f, indent=2)
