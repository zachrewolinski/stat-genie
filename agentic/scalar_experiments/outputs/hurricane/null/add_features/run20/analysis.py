import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data

df = pd.read_csv('hurricane.csv')

# Keep relevant columns
cols = ['alldeaths', 'masfem', 'gender_mf', 'min', 'wind', 'category', 'year', 'ndam15', 'ndam']
existing = [c for c in cols if c in df.columns]

d = df[existing].copy()

# Basic cleaning
for c in ['alldeaths', 'masfem', 'gender_mf', 'min', 'wind', 'category', 'year', 'ndam15', 'ndam']:
    if c in d.columns:
        d[c] = pd.to_numeric(d[c], errors='coerce')

d = d.dropna(subset=['alldeaths', 'masfem'])

# Outcome transformations

d['log_deaths'] = np.log1p(d['alldeaths'])

# Build regression formula manually using available controls
controls = []
for c in ['min', 'wind', 'category', 'year', 'ndam15']:
    if c in d.columns and d[c].notna().sum() > 0:
        controls.append(c)

# OLS with masfem

X_cols = ['masfem'] + controls
X = d[X_cols].dropna()
Y = d.loc[X.index, 'log_deaths']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

# Alternative with gender_mf
if 'gender_mf' in d.columns:
    X2_cols = ['gender_mf'] + controls
    X2 = d[X2_cols].dropna()
    Y2 = d.loc[X2.index, 'log_deaths']
    X2 = sm.add_constant(X2)
    model_gender = sm.OLS(Y2, X2).fit()
else:
    model_gender = None

# Simple correlation between masfem and log_deaths
corr = d[['masfem', 'log_deaths']].corr().iloc[0,1]

# Collect results
results = {
    'n_total': int(len(df)),
    'n_used_masfem': int(len(X)),
    'corr_masfem_log_deaths': float(corr),
    'masfem_coef': float(model.params['masfem']),
    'masfem_pval': float(model.pvalues['masfem']),
    'masfem_ci_low': float(model.conf_int().loc['masfem', 0]),
    'masfem_ci_high': float(model.conf_int().loc['masfem', 1]),
    'controls': controls,
}

if model_gender is not None and 'gender_mf' in model_gender.params:
    results.update({
        'gender_coef': float(model_gender.params['gender_mf']),
        'gender_pval': float(model_gender.pvalues['gender_mf']),
        'gender_ci_low': float(model_gender.conf_int().loc['gender_mf', 0]),
        'gender_ci_high': float(model_gender.conf_int().loc['gender_mf', 1]),
        'n_used_gender': int(len(model_gender.model.endog)),
    })

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
