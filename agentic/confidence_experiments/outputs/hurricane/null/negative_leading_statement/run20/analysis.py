import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Ensure numeric columns
num_cols = [
    'masfem', 'masfem_mturk', 'alldeaths', 'ndam', 'ndam15',
    'wind', 'min', 'category', 'year', 'gender_mf'
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Outcomes: deaths and log-deaths (add 1 to handle zeros)
df_log = df.copy()
df_log['log_deaths'] = np.log(df_log['alldeaths'] + 1)

# Define model formulas
formulas = {
    'ols_log_basic': 'log_deaths ~ masfem',
    'ols_log_controls': 'log_deaths ~ masfem + wind + min + category + year',
    'ols_log_controls_mturk': 'log_deaths ~ masfem_mturk + wind + min + category + year',
    'ols_log_gender': 'log_deaths ~ gender_mf + wind + min + category + year',
}

results = {}
for name, formula in formulas.items():
    model = smf.ols(formula, data=df_log).fit(cov_type='HC3')
    results[name] = {
        'n': int(model.nobs),
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'r2': float(model.rsquared),
    }

# Poisson GLM on deaths (counts) with log link; keep raw counts
# Include controls
glm_formula = 'alldeaths ~ masfem + wind + min + category + year'
try:
    glm_model = smf.glm(glm_formula, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
    results['glm_poisson'] = {
        'n': int(glm_model.nobs),
        'params': glm_model.params.to_dict(),
        'pvalues': glm_model.pvalues.to_dict(),
        'deviance': float(glm_model.deviance),
    }
except Exception as e:
    results['glm_poisson_error'] = str(e)

# Simple correlations
corrs = {}
for col in ['masfem', 'masfem_mturk', 'gender_mf']:
    if col in df_log.columns:
        corrs[col] = {
            'corr_log_deaths': float(df_log[[col, 'log_deaths']].corr().iloc[0, 1]),
            'corr_deaths': float(df_log[[col, 'alldeaths']].corr().iloc[0, 1]),
        }

output = {
    'results': results,
    'correlations': corrs,
    'summary': {
        'n_rows': int(df.shape[0]),
        'n_zero_deaths': int((df['alldeaths'] == 0).sum()),
    }
}

print(json.dumps(output, indent=2, sort_keys=True))
