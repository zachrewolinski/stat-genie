import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
file_path = 'hurricane.csv'
df = pd.read_csv(file_path)

# Basic cleaning
# Ensure numeric columns
numeric_cols = ['masfem','alldeaths','category','wind','min','ndam15','year']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# log deaths
# add 1 to handle zeros
df['log_deaths'] = np.log1p(df['alldeaths'])

# Simple correlation (Pearson) between masfem and log deaths
corr_pearson = df[['masfem','log_deaths']].corr().iloc[0,1]

# Spearman between masfem and alldeaths (raw)
corr_spearman = stats.spearmanr(df['masfem'], df['alldeaths'], nan_policy='omit')

# Model 1: log deaths ~ masfem
m1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# Model 2: log deaths ~ masfem + category + wind + min + year
# (controls for intensity and time)
m2 = smf.ols('log_deaths ~ masfem + category + wind + min + year', data=df).fit(cov_type='HC3')

# Model 3: add interaction masfem*category
m3 = smf.ols('log_deaths ~ masfem * category + wind + min + year', data=df).fit(cov_type='HC3')

# Extract key stats

def coef_info(model, term):
    coef = model.params.get(term, np.nan)
    pval = model.pvalues.get(term, np.nan)
    conf = model.conf_int().loc[term].tolist() if term in model.params.index else [np.nan, np.nan]
    return coef, pval, conf

results = {
    'n': int(df.shape[0]),
    'corr_pearson_masfem_logdeaths': float(corr_pearson),
    'corr_spearman_masfem_deaths': float(corr_spearman.correlation),
    'corr_spearman_p': float(corr_spearman.pvalue),
    'm1_masfem': coef_info(m1, 'masfem'),
    'm2_masfem': coef_info(m2, 'masfem'),
    'm3_masfem': coef_info(m3, 'masfem'),
    'm3_interaction': coef_info(m3, 'masfem:category'),
}

print('N', results['n'])
print('Pearson corr (masfem, log_deaths):', results['corr_pearson_masfem_logdeaths'])
print('Spearman corr (masfem, deaths):', results['corr_spearman_masfem_deaths'], 'p=', results['corr_spearman_p'])

print('\nModel 1 (log_deaths ~ masfem)')
coef, pval, conf = results['m1_masfem']
print('masfem coef', coef, 'p', pval, 'CI', conf)

print('\nModel 2 (log_deaths ~ masfem + category + wind + min + year)')
coef, pval, conf = results['m2_masfem']
print('masfem coef', coef, 'p', pval, 'CI', conf)

print('\nModel 3 (log_deaths ~ masfem * category + wind + min + year)')
coef, pval, conf = results['m3_masfem']
print('masfem coef', coef, 'p', pval, 'CI', conf)
coef, pval, conf = results['m3_interaction']
print('masfem:category coef', coef, 'p', pval, 'CI', conf)

