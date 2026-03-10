import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns to semantics based on observed ranges
# ndam: id
# wind: year
# alldeaths: name
# category: masfem index (1-11)
# ndam15: min pressure
# masfem_mturk: binary female
# gender_mf: Saffir-Simpson category
# name: deaths
# elapsedyrs: normalized damage (2013)
# masfem: years elapsed
# min: source
# ind: MTurk masfem rating
# year: max wind speed
# source: normalized damage (2015)


df = _df.copy()

# Outcome
# deaths

df['log_deaths'] = np.log1p(df['name'])

# Controls
controls = ['gender_mf', 'year', 'ndam15']  # category, max wind, min pressure

# Helper regression

def ols(y, X, data):
    X = data[X].copy()
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(data[y], X, missing='drop')
    res = model.fit()
    return res

results = {}

# Model 1: log deaths ~ masfem index + controls
res1 = ols('log_deaths', ['category'] + controls, df)
results['model1'] = res1

# Model 2: log deaths ~ mturk rating + controls
res2 = ols('log_deaths', ['ind'] + controls, df)
results['model2'] = res2

# Model 3: log deaths ~ binary female + controls
res3 = ols('log_deaths', ['masfem_mturk'] + controls, df)
results['model3'] = res3

# Model 4: add year of storm (wind) as control
res4 = ols('log_deaths', ['category'] + controls + ['wind'], df)
results['model4'] = res4

# Model 5: include damage (elapsedyrs) as control (may drop NAs)
res5 = ols('log_deaths', ['category'] + controls + ['elapsedyrs'], df)
results['model5'] = res5

# Correlations
corr = df[['category','ind','masfem_mturk','name','log_deaths']].corr(method='pearson')

# Spearman for robustness
spearman = df[['category','ind','masfem_mturk','name','log_deaths']].corr(method='spearman')

print('N', len(df))
print('\nPearson correlations (selected):')
print(corr[['log_deaths']].T)
print('\nSpearman correlations (selected):')
print(spearman[['log_deaths']].T)

print('\nModel summaries (key coef, p-value):')
for k, res in results.items():
    coef = res.params.get('category', np.nan)
    pval = res.pvalues.get('category', np.nan)
    coef_ind = res.params.get('ind', np.nan)
    pval_ind = res.pvalues.get('ind', np.nan)
    coef_bin = res.params.get('masfem_mturk', np.nan)
    pval_bin = res.pvalues.get('masfem_mturk', np.nan)
    print(k, 'n=', int(res.nobs))
    if not np.isnan(coef):
        print('  category coef', coef, 'p', pval)
    if not np.isnan(coef_ind):
        print('  ind coef', coef_ind, 'p', pval_ind)
    if not np.isnan(coef_bin):
        print('  masfem_mturk coef', coef_bin, 'p', pval_bin)

# Also model with deaths (not log) for sensitivity
res6 = ols('name', ['category'] + controls, df)
print('\nModel6 deaths ~ category + controls: coef', res6.params['category'], 'p', res6.pvalues['category'], 'n', int(res6.nobs))

# Save key stats for later

