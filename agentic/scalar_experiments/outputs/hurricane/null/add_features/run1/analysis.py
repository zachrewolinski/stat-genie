import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Keep relevant columns
cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'ndam', 'ndam15']
# Some columns might not exist; filter
cols = [c for c in cols if c in df.columns]

# Basic cleaning
# Drop rows with missing key fields
key_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year']
key_cols = [c for c in key_cols if c in df.columns]
clean = df.dropna(subset=key_cols).copy()

# Transform
clean['log_deaths'] = np.log1p(clean['alldeaths'])

results = {}

# Correlation
if clean['masfem'].std() > 0 and clean['log_deaths'].std() > 0:
    corr = clean['masfem'].corr(clean['log_deaths'])
else:
    corr = np.nan
results['corr_masfem_logdeaths'] = corr

# OLS models with robust SE
models = {}

# Model 1: bivariate
formula1 = 'log_deaths ~ masfem'
model1 = smf.ols(formula1, data=clean).fit(cov_type='HC3')
models['ols_bivariate'] = model1

# Model 2: controls for physical severity and year
formula2 = 'log_deaths ~ masfem + wind + min + category + year'
model2 = smf.ols(formula2, data=clean).fit(cov_type='HC3')
models['ols_controls'] = model2

# Model 3: alternative with gender_mf
if 'gender_mf' in clean.columns:
    formula3 = 'log_deaths ~ gender_mf + wind + min + category + year'
    model3 = smf.ols(formula3, data=clean).fit(cov_type='HC3')
    models['ols_gender_controls'] = model3

# Negative binomial (count model) with controls
# Use alldeaths as count; add 1 to avoid zeros? NB can handle zeros
try:
    formula_nb = 'alldeaths ~ masfem + wind + min + category + year'
    model_nb = smf.glm(formula_nb, data=clean, family=sm.families.NegativeBinomial()).fit()
    models['nb_controls'] = model_nb
except Exception as e:
    models['nb_controls'] = e

# Extract key stats

def extract(model, var='masfem'):
    if hasattr(model, 'params'):
        coef = model.params.get(var, np.nan)
        se = model.bse.get(var, np.nan) if hasattr(model, 'bse') else np.nan
        pval = model.pvalues.get(var, np.nan) if hasattr(model, 'pvalues') else np.nan
        return {'coef': coef, 'se': se, 'pval': pval}
    return None

summary = {
    'n': len(clean),
    'corr_masfem_logdeaths': corr,
    'ols_bivariate': extract(model1),
    'ols_controls': extract(model2),
}

if 'ols_gender_controls' in models:
    summary['ols_gender_controls'] = extract(models['ols_gender_controls'], var='gender_mf')

if isinstance(models.get('nb_controls'), Exception):
    summary['nb_controls_error'] = str(models['nb_controls'])
else:
    summary['nb_controls'] = extract(models['nb_controls'])

# Save for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Also print a readable summary
print(json.dumps(summary, indent=2))
