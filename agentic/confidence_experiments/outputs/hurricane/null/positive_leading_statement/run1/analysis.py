import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns
numeric_cols = ['masfem','min','category','alldeaths','ndam','elapsedyrs','masfem_mturk','wind','ndam15','gender_mf','year']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create log deaths (avoid log(0))
df['log_deaths'] = np.log1p(df['alldeaths'])

# Describe missingness
missing = df[numeric_cols + ['log_deaths']].isna().mean()

# Simple correlations
corr_masfem = df[['masfem','log_deaths']].corr().iloc[0,1]

# OLS regression: log_deaths on masfem + controls
# Controls: wind, min pressure, category, year, ndam15 (damage)
# Using ndam15 because normalized to 2015, and wind/min/category for intensity.
formula = 'log_deaths ~ masfem + wind + min + category + year + ndam15'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Alternative: log_deaths on gender_mf + controls
formula_g = 'log_deaths ~ gender_mf + wind + min + category + year + ndam15'
model_g = smf.ols(formula_g, data=df).fit(cov_type='HC3')

# Simpler model with only intensity controls (wind + min + category)
model_simple = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC3')

# Also test masfem_mturk as alternative femininity measure
model_mturk = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year + ndam15', data=df).fit(cov_type='HC3')

# Collect key stats

def coef_summary(m, var):
    return {
        'coef': float(m.params.get(var, np.nan)),
        'se': float(m.bse.get(var, np.nan)),
        'p': float(m.pvalues.get(var, np.nan)),
        'ci_low': float(m.conf_int().loc[var, 0]) if var in m.params.index else np.nan,
        'ci_high': float(m.conf_int().loc[var, 1]) if var in m.params.index else np.nan,
        'n': int(m.nobs)
    }

results = {
    'missing_rates': missing.to_dict(),
    'corr_masfem_log_deaths': float(corr_masfem),
    'ols_full_masfem': coef_summary(model, 'masfem'),
    'ols_full_gender': coef_summary(model_g, 'gender_mf'),
    'ols_simple_masfem': coef_summary(model_simple, 'masfem'),
    'ols_full_mturk': coef_summary(model_mturk, 'masfem_mturk'),
}

print(json.dumps(results, indent=2))
