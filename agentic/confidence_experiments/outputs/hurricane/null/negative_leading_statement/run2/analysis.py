import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic transforms
# log1p deaths to reduce skew
if 'alldeaths' in df.columns:
    df['log_deaths'] = np.log1p(df['alldeaths'])

# Ensure numeric columns

# Simple correlation between masfem and deaths
corr = df[['masfem','alldeaths']].corr().iloc[0,1]

# OLS: log_deaths ~ masfem + wind + min + category + year
# We'll use year (or elapsedyrs) to adjust time trends
# Some models may include interaction between masfem and severity; compute too.

# Build formula
formula = 'log_deaths ~ masfem + wind + min + category + year'
model = smf.ols(formula=formula, data=df).fit()

# Interaction model: masfem * wind (severity)
model_int = smf.ols(formula='log_deaths ~ masfem * wind + min + category + year', data=df).fit()

# Another model using gender_mf as binary
model_gender = smf.ols(formula='log_deaths ~ gender_mf + wind + min + category + year', data=df).fit()

# Output key stats
out = {
    'n': int(df.shape[0]),
    'corr_masfem_deaths': corr,
    'ols_main': {
        'coef_masfem': model.params.get('masfem', np.nan),
        'p_masfem': model.pvalues.get('masfem', np.nan),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
    },
    'ols_int': {
        'coef_masfem': model_int.params.get('masfem', np.nan),
        'p_masfem': model_int.pvalues.get('masfem', np.nan),
        'coef_masfem_wind': model_int.params.get('masfem:wind', np.nan),
        'p_masfem_wind': model_int.pvalues.get('masfem:wind', np.nan),
        'r2': model_int.rsquared,
        'adj_r2': model_int.rsquared_adj,
    },
    'ols_gender': {
        'coef_gender_mf': model_gender.params.get('gender_mf', np.nan),
        'p_gender_mf': model_gender.pvalues.get('gender_mf', np.nan),
        'r2': model_gender.rsquared,
        'adj_r2': model_gender.rsquared_adj,
    }
}

print(json.dumps(out, indent=2))
