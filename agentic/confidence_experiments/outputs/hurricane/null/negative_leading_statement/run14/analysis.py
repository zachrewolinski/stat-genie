import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import pingouin as pg

# Load data
df = pd.read_csv('hurricane.csv')

# Basic cleaning: ensure numeric columns
numeric_cols = ['masfem','masfem_mturk','gender_mf','category','wind','min','alldeaths','ndam','ndam15','year']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Derived outcomes
df['log_deaths'] = np.log1p(df['alldeaths'])

# Standardize for comparability (optional)
# but we will keep raw coefficients.

# Models
models = {}

# Simple bivariate relationships
models['bivar_masfem'] = smf.ols('log_deaths ~ masfem', data=df).fit()
models['bivar_gender'] = smf.ols('log_deaths ~ gender_mf', data=df).fit()

# Controls for severity and time
# Use wind and minimum pressure (min), category, year
models['controls_masfem'] = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit()
models['controls_gender'] = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit()

# Alternative: use masfem_mturk
if 'masfem_mturk' in df.columns:
    models['controls_masfem_mturk'] = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=df).fit()

# Count model (Poisson) for deaths
# Add 1 to avoid issues? Poisson can handle zeros.
try:
    models['poisson_masfem'] = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df, family=sm.families.Poisson()).fit()
    models['poisson_gender'] = smf.glm('alldeaths ~ gender_mf + wind + min + category + year', data=df, family=sm.families.Poisson()).fit()
    models['nb_masfem'] = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df, family=sm.families.NegativeBinomial()).fit()
    models['nb_gender'] = smf.glm('alldeaths ~ gender_mf + wind + min + category + year', data=df, family=sm.families.NegativeBinomial()).fit()
except Exception as e:
    models['count_error'] = e

# Summaries of key coefficients
def extract_coef(model):
    params = model.params
    if 'masfem' in params.index:
        return 'masfem'
    if 'gender_mf' in params.index:
        return 'gender_mf'
    if 'masfem_mturk' in params.index:
        return 'masfem_mturk'
    return None

results = {}
for name, model in models.items():
    if hasattr(model, 'params'):
        coef_name = extract_coef(model)
        if coef_name:
            # robust SE for OLS models
            if name.startswith('bivar') or name.startswith('controls'):
                robust = model.get_robustcov_results(cov_type='HC3')
                pval = float(robust.pvalues[model.params.index.get_loc(coef_name)])
            else:
                pval = float(model.pvalues[coef_name])
            results[name] = {
                'coef': float(model.params[coef_name]),
                'pvalue': pval,
                'n': int(model.nobs),
                'r2': float(getattr(model, 'rsquared', np.nan))
            }

# Descriptives
summary = {
    'n': int(df.shape[0]),
    'deaths_mean': float(df['alldeaths'].mean()),
    'deaths_median': float(df['alldeaths'].median()),
    'deaths_zero_share': float((df['alldeaths'] == 0).mean()),
    'deaths_var': float(df['alldeaths'].var()),
    'masfem_mean': float(df['masfem'].mean()),
    'masfem_sd': float(df['masfem'].std()),
}

# Correlations
pearson = stats.pearsonr(df['masfem'], df['log_deaths'])
spearman = stats.spearmanr(df['masfem'], df['log_deaths'])

partial = pg.partial_corr(data=df, x='masfem', y='log_deaths', covar=['wind','min','category','year'], method='pearson')
partial_r = float(partial['r'].iloc[0])
partial_p = float(partial['p-val'].iloc[0])

correlations = {
    'pearson_r': float(pearson.statistic),
    'pearson_p': float(pearson.pvalue),
    'spearman_r': float(spearman.statistic),
    'spearman_p': float(spearman.pvalue),
    'partial_r': partial_r,
    'partial_p': partial_p
}

print('SUMMARY', summary)
print('CORRELATIONS', correlations)
print('RESULTS')
for k, v in results.items():
    print(k, v)

# Save results to a file for later use
import json
with open('analysis_results.json', 'w') as f:
    json.dump({'summary': summary, 'correlations': correlations, 'results': results}, f, indent=2)
