import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Keep relevant columns; handle missing
cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'ndam', 'ndam15', 'year']
existing_cols = [c for c in cols if c in df.columns]

# Coerce numeric where needed
for c in existing_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with missing key outcome or predictors
base_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category']
base_cols = [c for c in base_cols if c in df.columns]
analysis_df = df[base_cols + [c for c in ['ndam','ndam15','year'] if c in df.columns]].dropna()

# Outcome transformations
analysis_df['log_deaths'] = np.log1p(analysis_df['alldeaths'])

# Simple correlations
corr_masfem = analysis_df[['masfem', 'alldeaths', 'log_deaths']].corr().loc['masfem']

# OLS on log_deaths with controls
# Use wind, min, category as intensity controls (all numeric)
formula = 'log_deaths ~ masfem + wind + min + category'
ols = smf.ols(formula, data=analysis_df).fit(cov_type='HC3')

# OLS with gender_mf if available
ols_gender = None
if 'gender_mf' in analysis_df.columns:
    formula_gender = 'log_deaths ~ gender_mf + wind + min + category'
    ols_gender = smf.ols(formula_gender, data=analysis_df).fit(cov_type='HC3')

# Negative binomial on death counts (if possible)
# Add small constant to avoid issues; use NB with log link
nb = smf.glm('alldeaths ~ masfem + wind + min + category', data=analysis_df,
             family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

# Summaries to json
summary = {
    'n': int(len(analysis_df)),
    'corr_masfem_alldeaths': float(corr_masfem['alldeaths']),
    'corr_masfem_log_deaths': float(corr_masfem['log_deaths']),
    'ols_masfem_coef': float(ols.params['masfem']),
    'ols_masfem_p': float(ols.pvalues['masfem']),
    'ols_masfem_ci': [float(x) for x in ols.conf_int().loc['masfem']],
    'ols_r2': float(ols.rsquared),
    'nb_masfem_coef': float(nb.params['masfem']),
    'nb_masfem_p': float(nb.pvalues['masfem']),
    'nb_masfem_ci': [float(x) for x in nb.conf_int().loc['masfem']],
}
if ols_gender is not None:
    summary.update({
        'ols_gender_coef': float(ols_gender.params['gender_mf']),
        'ols_gender_p': float(ols_gender.pvalues['gender_mf']),
        'ols_gender_ci': [float(x) for x in ols_gender.conf_int().loc['gender_mf']],
    })

print(json.dumps(summary, indent=2))
