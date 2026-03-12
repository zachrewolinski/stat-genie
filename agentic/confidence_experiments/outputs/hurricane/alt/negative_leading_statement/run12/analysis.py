import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [
    'masfem','masfem_mturk','gender_mf','alldeaths','ndam','ndam15','wind','min','category','year'
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create log outcomes
df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_dam15'] = np.log1p(df['ndam15'])

# Drop rows with missing critical variables
base_cols = ['masfem','gender_mf','log_deaths','log_dam15','wind','min','category','year']
df_model = df[base_cols].dropna().copy()

# Helper to fit model and extract coef/p

def fit_model(formula, data):
    model = smf.ols(formula, data=data).fit(cov_type='HC3')
    return {
        'formula': formula,
        'n': int(model.nobs),
        'coef': model.params.to_dict(),
        'pvals': model.pvalues.to_dict(),
        'r2': float(model.rsquared),
    }

results = {}

# Bivariate relationships
results['bivar_deaths_masfem'] = fit_model('log_deaths ~ masfem', df_model)
results['bivar_dam_masfem'] = fit_model('log_dam15 ~ masfem', df_model)
results['bivar_deaths_gender'] = fit_model('log_deaths ~ gender_mf', df_model)
results['bivar_dam_gender'] = fit_model('log_dam15 ~ gender_mf', df_model)

# Controlled models: intensity controls
controls = 'wind + min + category + year'
results['ctrl_deaths_masfem'] = fit_model(f'log_deaths ~ masfem + {controls}', df_model)
results['ctrl_dam_masfem'] = fit_model(f'log_dam15 ~ masfem + {controls}', df_model)
results['ctrl_deaths_gender'] = fit_model(f'log_deaths ~ gender_mf + {controls}', df_model)
results['ctrl_dam_gender'] = fit_model(f'log_dam15 ~ gender_mf + {controls}', df_model)

# Alternative: use masfem_mturk (if available)
if 'masfem_mturk' in df.columns:
    df_model2 = df[['masfem_mturk','log_deaths','log_dam15','wind','min','category','year']].dropna()
    results['ctrl_deaths_mturk'] = fit_model(f'log_deaths ~ masfem_mturk + {controls}', df_model2)
    results['ctrl_dam_mturk'] = fit_model(f'log_dam15 ~ masfem_mturk + {controls}', df_model2)

# Correlations
corrs = {}
for col in ['masfem','gender_mf','masfem_mturk']:
    if col in df.columns:
        corrs[col] = {
            'log_deaths': float(df[[col,'log_deaths']].corr().iloc[0,1]),
            'log_dam15': float(df[[col,'log_dam15']].corr().iloc[0,1])
        }
results['correlations'] = corrs

print(json.dumps(results, indent=2))
