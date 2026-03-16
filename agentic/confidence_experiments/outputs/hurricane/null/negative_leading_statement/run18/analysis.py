import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Prepare variables
_df = df.copy()
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Basic correlation (drop missing)
_corr_df = _df[['masfem', 'log_deaths']].dropna()
corr = stats.pearsonr(_corr_df['masfem'], _corr_df['log_deaths'])

# Helper to fit model with dropna

def fit_ols(endog_col, exog_cols):
    data = _df[[endog_col] + exog_cols].dropna().copy()
    y = data[endog_col]
    X = sm.add_constant(data[exog_cols])
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model, len(data)

# Regression with controls
model, n1 = fit_ols('log_deaths', ['masfem', 'wind', 'min', 'category', 'ndam', 'year'])
model2, n2 = fit_ols('log_deaths', ['masfem', 'wind', 'min', 'category', 'ndam15', 'year'])
model3, n3 = fit_ols('log_deaths', ['gender_mf', 'wind', 'min', 'category', 'ndam', 'year'])

# Interaction model: masfem * wind
_df['masfem_x_wind'] = _df['masfem'] * _df['wind']
model4, n4 = fit_ols('log_deaths', ['masfem', 'wind', 'masfem_x_wind', 'min', 'category', 'ndam', 'year'])

# Save summary stats
result = {
    'n_corr': int(len(_corr_df)),
    'corr_r': float(corr.statistic),
    'corr_p': float(corr.pvalue),
    'n_model': int(n1),
    'model_masfem_coef': float(model.params['masfem']),
    'model_masfem_p': float(model.pvalues['masfem']),
    'model_masfem_ci': [float(x) for x in model.conf_int().loc['masfem'].tolist()],
    'n_model2': int(n2),
    'model2_masfem_coef': float(model2.params['masfem']),
    'model2_masfem_p': float(model2.pvalues['masfem']),
    'model2_masfem_ci': [float(x) for x in model2.conf_int().loc['masfem'].tolist()],
    'n_model3': int(n3),
    'model3_gender_coef': float(model3.params['gender_mf']),
    'model3_gender_p': float(model3.pvalues['gender_mf']),
    'model3_gender_ci': [float(x) for x in model3.conf_int().loc['gender_mf'].tolist()],
    'n_model4': int(n4),
    'model4_interaction_coef': float(model4.params['masfem_x_wind']),
    'model4_interaction_p': float(model4.pvalues['masfem_x_wind']),
    'model4_interaction_ci': [float(x) for x in model4.conf_int().loc['masfem_x_wind'].tolist()],
}

print(result)
