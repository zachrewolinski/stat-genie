import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Drop rows with missing key covariates
key_cols = ['alldeaths','masfem','wind','min','category','year']
df_clean = df.dropna(subset=key_cols).copy()

# Outcome transformations
# Log1p to handle zeros and skew
# Also prepare for GLM counts

df_clean['log_deaths'] = np.log1p(df_clean['alldeaths'])

# Correlations
pearson = stats.pearsonr(df_clean['masfem'], df_clean['log_deaths'])
spearman = stats.spearmanr(df_clean['masfem'], df_clean['log_deaths'])

# Dispersion check for count outcome
mean_deaths = df_clean['alldeaths'].mean()
var_deaths = df_clean['alldeaths'].var(ddof=1)

# OLS with robust SEs
formula_ols = 'log_deaths ~ masfem + wind + min + category + year'
ols_model = smf.ols(formula_ols, data=df_clean).fit(cov_type='HC3')

formula_ols_gender = 'log_deaths ~ gender_mf + wind + min + category + year'
ols_gender = smf.ols(formula_ols_gender, data=df_clean).fit(cov_type='HC3')

# Negative binomial GLM
formula_nb = 'alldeaths ~ masfem + wind + min + category + year'
nb_model = smf.glm(formula_nb, data=df_clean, family=sm.families.NegativeBinomial()).fit()

formula_nb_gender = 'alldeaths ~ gender_mf + wind + min + category + year'
nb_gender = smf.glm(formula_nb_gender, data=df_clean, family=sm.families.NegativeBinomial()).fit()

# Poisson GLM (for comparison)
formula_pois = 'alldeaths ~ masfem + wind + min + category + year'
pois_model = smf.glm(formula_pois, data=df_clean, family=sm.families.Poisson()).fit()

formula_pois_gender = 'alldeaths ~ gender_mf + wind + min + category + year'
pois_gender = smf.glm(formula_pois_gender, data=df_clean, family=sm.families.Poisson()).fit()

# Discrete Negative Binomial (estimates dispersion)
exog_cols = ['masfem', 'wind', 'min', 'category', 'year']
exog_m = sm.add_constant(df_clean[exog_cols])
exog_g = sm.add_constant(df_clean[['gender_mf', 'wind', 'min', 'category', 'year']])
endog = df_clean['alldeaths']

nb2_m = sm.NegativeBinomial(endog, exog_m).fit(disp=False)
nb2_g = sm.NegativeBinomial(endog, exog_g).fit(disp=False)

# Summaries
result = {
    'n_rows': int(df_clean.shape[0]),
    'pearson_r': pearson[0],
    'pearson_p': pearson[1],
    'spearman_rho': spearman.correlation,
    'spearman_p': spearman.pvalue,
    'mean_deaths': mean_deaths,
    'var_deaths': var_deaths,
    'ols_masfem_coef': ols_model.params['masfem'],
    'ols_masfem_p': ols_model.pvalues['masfem'],
    'ols_masfem_ci': tuple(ols_model.conf_int().loc['masfem']),
    'ols_gender_coef': ols_gender.params['gender_mf'],
    'ols_gender_p': ols_gender.pvalues['gender_mf'],
    'ols_gender_ci': tuple(ols_gender.conf_int().loc['gender_mf']),
    'nb_masfem_coef': nb_model.params['masfem'],
    'nb_masfem_p': nb_model.pvalues['masfem'],
    'nb_masfem_ci': tuple(nb_model.conf_int().loc['masfem']),
    'nb_gender_coef': nb_gender.params['gender_mf'],
    'nb_gender_p': nb_gender.pvalues['gender_mf'],
    'nb_gender_ci': tuple(nb_gender.conf_int().loc['gender_mf']),
    'pois_masfem_coef': pois_model.params['masfem'],
    'pois_masfem_p': pois_model.pvalues['masfem'],
    'pois_gender_coef': pois_gender.params['gender_mf'],
    'pois_gender_p': pois_gender.pvalues['gender_mf'],
    'nb2_masfem_coef': nb2_m.params[exog_m.columns.get_loc('masfem')],
    'nb2_masfem_p': nb2_m.pvalues[exog_m.columns.get_loc('masfem')],
    'nb2_masfem_ci': tuple(nb2_m.conf_int().loc['masfem']),
    'nb2_gender_coef': nb2_g.params[exog_g.columns.get_loc('gender_mf')],
    'nb2_gender_p': nb2_g.pvalues[exog_g.columns.get_loc('gender_mf')],
    'nb2_gender_ci': tuple(nb2_g.conf_int().loc['gender_mf']),
    'nb2_alpha_m': nb2_m.params[exog_m.shape[1]],  # last param is alpha
    'nb2_alpha_g': nb2_g.params[exog_g.shape[1]],
}

# Also model with interaction? maybe severity interaction; compute if needed
# Common in literature: interaction with severity (e.g., wind). We'll compute interaction with wind.
formula_ols_inter = 'log_deaths ~ masfem * wind + min + category + year'
ols_inter = smf.ols(formula_ols_inter, data=df_clean).fit(cov_type='HC3')

result.update({
    'ols_inter_masfem_coef': ols_inter.params['masfem'],
    'ols_inter_masfem_p': ols_inter.pvalues['masfem'],
    'ols_inter_masfem_wind_coef': ols_inter.params['masfem:wind'],
    'ols_inter_masfem_wind_p': ols_inter.pvalues['masfem:wind'],
})

# Save results for inspection
import json
with open('analysis_results.json','w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
