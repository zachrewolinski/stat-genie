import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

_df = pd.read_csv('hurricane.csv')
for col in ['masfem','masfem_mturk','min','wind','category','alldeaths','ndam','ndam15','gender_mf']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')
_df = _df.dropna(subset=['alldeaths','masfem','wind','min','category'])
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Additional controls: ndam15 if available
controls = 'wind + min + category'
if 'ndam15' in _df.columns:
    controls2 = controls + ' + ndam15'
else:
    controls2 = controls

# Poisson with robust SE
poisson = smf.glm(f'alldeaths ~ masfem + {controls}', data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')

# NegBin with robust SE
nb = smf.glm(f'alldeaths ~ masfem + {controls}', data=_df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

# OLS on log deaths with additional controls
ols = smf.ols(f'log_deaths ~ masfem + {controls}', data=_df).fit(cov_type='HC3')

# OLS on log deaths with ndam15
ols2 = smf.ols(f'log_deaths ~ masfem + {controls2}', data=_df).fit(cov_type='HC3')

# Alternative masfem_mturk
if 'masfem_mturk' in _df.columns:
    ols_mturk = smf.ols(f'log_deaths ~ masfem_mturk + {controls}', data=_df).fit(cov_type='HC3')
else:
    ols_mturk = None

# Outlier check: remove top 1% deaths or top 3 deaths
_df_sorted = _df.sort_values('alldeaths', ascending=False)
_df_trim = _df_sorted.iloc[3:].copy()
ols_trim = smf.ols(f'log_deaths ~ masfem + {controls}', data=_df_trim).fit(cov_type='HC3')

print('Poisson masfem coef/p:', poisson.params['masfem'], poisson.pvalues['masfem'])
print('NB masfem coef/p (robust):', nb.params['masfem'], nb.pvalues['masfem'])
print('OLS log deaths masfem coef/p:', ols.params['masfem'], ols.pvalues['masfem'])
print('OLS log deaths + ndam15 masfem coef/p:', ols2.params['masfem'], ols2.pvalues['masfem'])
if ols_mturk is not None:
    print('OLS log deaths masfem_mturk coef/p:', ols_mturk.params['masfem_mturk'], ols_mturk.pvalues['masfem_mturk'])
print('OLS log deaths trimmed top3 deaths masfem coef/p:', ols_trim.params['masfem'], ols_trim.pvalues['masfem'])

# Save for inspection
results = {
    'poisson': {'coef': float(poisson.params['masfem']), 'p': float(poisson.pvalues['masfem'])},
    'nb_robust': {'coef': float(nb.params['masfem']), 'p': float(nb.pvalues['masfem'])},
    'ols': {'coef': float(ols.params['masfem']), 'p': float(ols.pvalues['masfem'])},
    'ols_ndam15': {'coef': float(ols2.params['masfem']), 'p': float(ols2.pvalues['masfem'])},
    'ols_trim_top3': {'coef': float(ols_trim.params['masfem']), 'p': float(ols_trim.pvalues['masfem'])},
}
if ols_mturk is not None:
    results['ols_mturk'] = {'coef': float(ols_mturk.params['masfem_mturk']), 'p': float(ols_mturk.pvalues['masfem_mturk'])}

pd.DataFrame(results).to_csv('analysis_results_extra.csv')
