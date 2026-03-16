import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('hurricane.csv')

# Basic preprocessing
# Log-transform outcomes due to skew; add 1 to handle zeros

df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam15'] = np.log1p(df['ndam15'])

# Core variables

masfem = df['masfem']

# Correlations

corr_pearson = stats.pearsonr(masfem, df['log_deaths'])
corr_spearman = stats.spearmanr(masfem, df['log_deaths'])

# Simple OLS: log deaths ~ masfem
model_simple = smf.ols('log_deaths ~ masfem', data=df).fit()

# Controls for storm severity
# Use wind and min pressure and category; also control for year (long-run trends)
model_controls = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit()

# Alternative: include damage as a proxy for exposure/impact
model_controls_damage = smf.ols('log_deaths ~ masfem + wind + min + category + year + log_ndam15', data=df).fit()

# Binary gender indicator as robustness
model_gender = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit()

# Save key outputs

results = {
    'n': len(df),
    'corr_pearson_r': corr_pearson.statistic,
    'corr_pearson_p': corr_pearson.pvalue,
    'corr_spearman_r': corr_spearman.statistic,
    'corr_spearman_p': corr_spearman.pvalue,
    'simple_coef_masfem': model_simple.params.get('masfem'),
    'simple_p_masfem': model_simple.pvalues.get('masfem'),
    'controls_coef_masfem': model_controls.params.get('masfem'),
    'controls_p_masfem': model_controls.pvalues.get('masfem'),
    'controls_damage_coef_masfem': model_controls_damage.params.get('masfem'),
    'controls_damage_p_masfem': model_controls_damage.pvalues.get('masfem'),
    'gender_coef': model_gender.params.get('gender_mf'),
    'gender_p': model_gender.pvalues.get('gender_mf'),
    'simple_r2': model_simple.rsquared,
    'controls_r2': model_controls.rsquared,
    'controls_damage_r2': model_controls_damage.rsquared,
}

# Print for inspection

print(results)
