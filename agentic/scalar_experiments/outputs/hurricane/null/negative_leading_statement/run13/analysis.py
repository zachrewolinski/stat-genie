import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA = Path('hurricane.csv')

df = pd.read_csv(DATA)

# Basic prep
# log deaths

df['log_alldeaths'] = np.log1p(df['alldeaths'])

# Core variables
# masfem (higher = more feminine)

# Simple correlations
corr = df['masfem'].corr(df['alldeaths'])
corr_log = df['masfem'].corr(df['log_alldeaths'])

# OLS: log deaths ~ masfem + controls (wind, min pressure, category, ndam15)
# Use log of damages because skew

df['log_ndam15'] = np.log1p(df['ndam15'])

formula = 'log_alldeaths ~ masfem + wind + min + category + log_ndam15'
model = smf.ols(formula, data=df).fit()

# Alternative: use gender_mf binary instead of continuous
formula2 = 'log_alldeaths ~ gender_mf + wind + min + category + log_ndam15'
model2 = smf.ols(formula2, data=df).fit()

# Bivariate regression without controls
model_simple = smf.ols('log_alldeaths ~ masfem', data=df).fit()

# Spearman correlation for robustness
spearman = df['masfem'].corr(df['alldeaths'], method='spearman')

# Collect key results
results = {
    'n': len(df),
    'corr_masfem_alldeaths': corr,
    'corr_masfem_log_alldeaths': corr_log,
    'spearman_masfem_alldeaths': spearman,
    'simple_coef_masfem': model_simple.params['masfem'],
    'simple_p_masfem': model_simple.pvalues['masfem'],
    'control_coef_masfem': model.params['masfem'],
    'control_p_masfem': model.pvalues['masfem'],
    'control_coef_gender_mf': model2.params['gender_mf'],
    'control_p_gender_mf': model2.pvalues['gender_mf'],
}

print(json.dumps(results, indent=2))
