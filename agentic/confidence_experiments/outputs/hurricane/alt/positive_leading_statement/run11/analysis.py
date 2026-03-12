import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.discrete.discrete_model as smd

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic derived variables
_df['log_deaths'] = np.log1p(_df['alldeaths'])
_df['log_ndam15'] = np.log1p(_df['ndam15'])

# Drop rows with missing in key vars
key_vars = ['masfem', 'alldeaths', 'log_deaths', 'wind', 'min', 'category', 'log_ndam15', 'year']
_df_model = _df.dropna(subset=key_vars).copy()

# Correlations
corr_pearson = _df_model[['masfem', 'alldeaths', 'log_deaths']].corr(method='pearson')
corr_spearman = _df_model[['masfem', 'alldeaths', 'log_deaths']].corr(method='spearman')

# OLS: log deaths
ols1 = smf.ols('log_deaths ~ masfem', data=_df_model).fit()
ols2 = smf.ols('log_deaths ~ masfem + wind + min + category + log_ndam15 + year', data=_df_model).fit()

# Poisson GLM: deaths (count) with log link
# Add small offset? not needed. Use robust SE due to overdispersion possibility
poisson = smf.glm('alldeaths ~ masfem + wind + min + category + log_ndam15 + year',
                  data=_df_model, family=sm.families.Poisson()).fit(cov_type='HC3')

# Negative Binomial (discrete) to estimate overdispersion
exog = sm.add_constant(_df_model[['masfem', 'wind', 'min', 'category', 'log_ndam15', 'year']])
endog = _df_model['alldeaths']
nb2 = smd.NegativeBinomial(endog, exog).fit(disp=0)

# Summaries for key coefficient
summary = {
    'n': int(_df_model.shape[0]),
    'corr_pearson_masfem_alldeaths': corr_pearson.loc['masfem', 'alldeaths'],
    'corr_pearson_masfem_logdeaths': corr_pearson.loc['masfem', 'log_deaths'],
    'corr_spearman_masfem_alldeaths': corr_spearman.loc['masfem', 'alldeaths'],
    'corr_spearman_masfem_logdeaths': corr_spearman.loc['masfem', 'log_deaths'],
    'ols1_coef': ols1.params['masfem'],
    'ols1_pval': ols1.pvalues['masfem'],
    'ols2_coef': ols2.params['masfem'],
    'ols2_pval': ols2.pvalues['masfem'],
    'poisson_coef': poisson.params['masfem'],
    'poisson_pval': poisson.pvalues['masfem'],
    'nb2_coef': nb2.params['masfem'],
    'nb2_pval': nb2.pvalues['masfem'],
}

# Also capture confidence intervals
summary['ols2_ci_low'], summary['ols2_ci_high'] = ols2.conf_int().loc['masfem']
summary['poisson_ci_low'], summary['poisson_ci_high'] = poisson.conf_int().loc['masfem']
summary['nb2_ci_low'], summary['nb2_ci_high'] = nb2.conf_int().loc['masfem']

# Save outputs for inspection
pd.Series(summary).to_csv('analysis_summary.csv')

# Print a compact report
print('N:', summary['n'])
print('Pearson corr (masfem, alldeaths):', summary['corr_pearson_masfem_alldeaths'])
print('Pearson corr (masfem, log_deaths):', summary['corr_pearson_masfem_logdeaths'])
print('Spearman corr (masfem, alldeaths):', summary['corr_spearman_masfem_alldeaths'])
print('Spearman corr (masfem, log_deaths):', summary['corr_spearman_masfem_logdeaths'])
print('OLS1 masfem coef, p:', summary['ols1_coef'], summary['ols1_pval'])
print('OLS2 masfem coef, p:', summary['ols2_coef'], summary['ols2_pval'])
print('OLS2 masfem CI:', (summary['ols2_ci_low'], summary['ols2_ci_high']))
print('Poisson masfem coef, p:', summary['poisson_coef'], summary['poisson_pval'])
print('Poisson masfem CI:', (summary['poisson_ci_low'], summary['poisson_ci_high']))
print('NegBin2 masfem coef, p:', summary['nb2_coef'], summary['nb2_pval'])
print('NegBin2 masfem CI:', (summary['nb2_ci_low'], summary['nb2_ci_high']))
