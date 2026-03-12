import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Prepare variables
# Log transform outcomes to reduce skew and handle zeros
_df['log_deaths'] = np.log1p(_df['alldeaths'])
_df['log_ndam'] = np.log1p(_df['ndam'])
_df['log_ndam15'] = np.log1p(_df['ndam15'])

# Core predictors
# Use masfem (continuous femininity) and gender_mf (binary) in separate models
# Controls: intensity measures (wind, min, category), year (or elapsedyrs)

# Drop rows with missing data in relevant columns
vars_core = ['log_deaths', 'masfem', 'wind', 'min', 'category', 'year']
_df_core = _df.dropna(subset=vars_core)

X = _df_core[['masfem', 'wind', 'min', 'category', 'year']]
X = sm.add_constant(X)
model = sm.OLS(_df_core['log_deaths'], X).fit(cov_type='HC3')

# Binary gender model
vars_bin = ['log_deaths', 'gender_mf', 'wind', 'min', 'category', 'year']
_df_bin = _df.dropna(subset=vars_bin)
Xb = _df_bin[['gender_mf', 'wind', 'min', 'category', 'year']]
Xb = sm.add_constant(Xb)
model_bin = sm.OLS(_df_bin['log_deaths'], Xb).fit(cov_type='HC3')

# Damage outcome as supplementary evidence
vars_dam = ['log_ndam', 'masfem', 'wind', 'min', 'category', 'year']
_df_dam = _df.dropna(subset=vars_dam)
Xd = _df_dam[['masfem', 'wind', 'min', 'category', 'year']]
Xd = sm.add_constant(Xd)
model_dam = sm.OLS(_df_dam['log_ndam'], Xd).fit(cov_type='HC3')

# Summaries for reporting
results = {
    'n_deaths': int(_df_core.shape[0]),
    'coef_masfem': model.params['masfem'],
    'se_masfem': model.bse['masfem'],
    'p_masfem': model.pvalues['masfem'],
    'coef_gender_mf': model_bin.params['gender_mf'],
    'se_gender_mf': model_bin.bse['gender_mf'],
    'p_gender_mf': model_bin.pvalues['gender_mf'],
    'coef_masfem_damage': model_dam.params['masfem'],
    'se_masfem_damage': model_dam.bse['masfem'],
    'p_masfem_damage': model_dam.pvalues['masfem'],
}

print(json.dumps(results, indent=2))
