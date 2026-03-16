import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('hurricane.csv')

# Basic preprocessing
# Use log1p for deaths due to heavy skew

df['log_deaths'] = np.log1p(df['alldeaths'])

# Define covariates (storm intensity + time)
# Some storms may have missing values; drop rows with NA in model columns
model_cols = ['log_deaths', 'masfem', 'category', 'wind', 'min', 'year']
model_df = df[model_cols].dropna()

X = model_df[['masfem', 'category', 'wind', 'min', 'year']]
X = sm.add_constant(X)

# OLS with robust (HC3) standard errors
model = sm.OLS(model_df['log_deaths'], X).fit(cov_type='HC3')

# Also check binary gender indicator for comparison
model_cols2 = ['log_deaths', 'gender_mf', 'category', 'wind', 'min', 'year']
model_df2 = df[model_cols2].dropna()
X2 = sm.add_constant(model_df2[['gender_mf', 'category', 'wind', 'min', 'year']])
model2 = sm.OLS(model_df2['log_deaths'], X2).fit(cov_type='HC3')

# Simple correlations (Spearman and Pearson)
pearson = df[['masfem', 'alldeaths']].corr(method='pearson').iloc[0,1]
spearman = df[['masfem', 'alldeaths']].corr(method='spearman').iloc[0,1]

# Build concise results
results = {
    'n_total': int(len(df)),
    'n_model': int(len(model_df)),
    'corr_pearson_masfem_deaths': float(pearson),
    'corr_spearman_masfem_deaths': float(spearman),
    'model_coef_masfem': float(model.params['masfem']),
    'model_pval_masfem': float(model.pvalues['masfem']),
    'model_ci_masfem': [float(x) for x in model.conf_int().loc['masfem']],
    'model_r2': float(model.rsquared),
    'model2_coef_gender_mf': float(model2.params['gender_mf']),
    'model2_pval_gender_mf': float(model2.pvalues['gender_mf']),
    'model2_ci_gender_mf': [float(x) for x in model2.conf_int().loc['gender_mf']],
    'model2_r2': float(model2.rsquared),
}

print(json.dumps(results, indent=2))
