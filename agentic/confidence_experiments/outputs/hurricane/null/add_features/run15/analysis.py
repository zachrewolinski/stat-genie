import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('hurricane.csv')

# Select relevant columns and drop rows with missing key values
cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'ndam', 'ndam15', 'year']
# Some columns may have missing values
_df = _df[cols].copy()

# Create derived variables
_df['log_deaths'] = np.log1p(_df['alldeaths'])
_df['log_ndam'] = np.log1p(_df['ndam'])
_df['log_ndam15'] = np.log1p(_df['ndam15'])

# Drop rows with any missing values in key vars for regression
reg_vars = ['log_deaths', 'masfem', 'wind', 'min', 'log_ndam', 'year']
reg_df = _df.dropna(subset=reg_vars).copy()

# Basic correlations
pearson_r, pearson_p = stats.pearsonr(reg_df['masfem'], reg_df['log_deaths'])
spearman_r, spearman_p = stats.spearmanr(reg_df['masfem'], reg_df['log_deaths'])

# Model 1: main effects
model1 = smf.ols('log_deaths ~ masfem + wind + min + log_ndam + year', data=reg_df).fit(cov_type='HC3')

# Model 2: interaction with storm severity (wind)
model2 = smf.ols('log_deaths ~ masfem * wind + min + log_ndam + year', data=reg_df).fit(cov_type='HC3')

# Model 3: interaction with damage (proxy for intensity/exposure)
model3 = smf.ols('log_deaths ~ masfem * log_ndam + wind + min + year', data=reg_df).fit(cov_type='HC3')

# Also check binary gender effect
model4 = smf.ols('log_deaths ~ gender_mf + wind + min + log_ndam + year', data=reg_df).fit(cov_type='HC3')

# Prepare a compact summary
print('N used:', len(reg_df))
print('Pearson r (masfem vs log_deaths):', pearson_r, 'p=', pearson_p)
print('Spearman r:', spearman_r, 'p=', spearman_p)

for name, m in [('model1', model1), ('model2', model2), ('model3', model3), ('model4', model4)]:
    coef = m.params.get('masfem', np.nan)
    pval = m.pvalues.get('masfem', np.nan)
    if name == 'model2':
        inter = m.params.get('masfem:wind', np.nan)
        inter_p = m.pvalues.get('masfem:wind', np.nan)
        print(name, 'masfem coef', coef, 'p', pval, 'interaction', inter, 'p', inter_p)
    elif name == 'model3':
        inter = m.params.get('masfem:log_ndam', np.nan)
        inter_p = m.pvalues.get('masfem:log_ndam', np.nan)
        print(name, 'masfem coef', coef, 'p', pval, 'interaction', inter, 'p', inter_p)
    else:
        print(name, 'masfem coef', coef, 'p', pval)

# Print gender_mf model
print('model4 gender_mf coef', model4.params.get('gender_mf', np.nan), 'p', model4.pvalues.get('gender_mf', np.nan))
