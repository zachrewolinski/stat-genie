import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('hurricane.csv')

# Create log deaths
df['log_deaths'] = np.log1p(df['alldeaths'])

# Basic correlation
corr = df[['masfem', 'log_deaths', 'alldeaths']].corr(method='spearman')

# Models
models = {}

# Unadjusted
models['unadjusted'] = smf.ols('log_deaths ~ masfem', data=df).fit()

# Adjust for intensity (wind, min pressure, category)
models['intensity'] = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit()

# Adjust intensity + year + damage (ndam15)
models['intensity_year_damage'] = smf.ols('log_deaths ~ masfem + wind + min + category + year + ndam15', data=df).fit()

# Use binary gender_mf too
models['gender_mf_intensity'] = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=df).fit()

# Print summary stats
print('Spearman correlations:\n', corr)

for name, model in models.items():
    print('\nModel:', name)
    print(model.summary().tables[1])

# Save key outputs to a JSON-ish dict for parsing
out = {
    'spearman_masfem_log_deaths': float(corr.loc['masfem', 'log_deaths']),
    'spearman_masfem_alldeaths': float(corr.loc['masfem', 'alldeaths']),
}

for name, model in models.items():
    coef = model.params.get('masfem', np.nan)
    pval = model.pvalues.get('masfem', np.nan)
    coef_g = model.params.get('gender_mf', np.nan)
    pval_g = model.pvalues.get('gender_mf', np.nan)
    out[name] = {
        'coef_masfem': float(coef) if coef == coef else None,
        'p_masfem': float(pval) if pval == pval else None,
        'coef_gender_mf': float(coef_g) if coef_g == coef_g else None,
        'p_gender_mf': float(pval_g) if pval_g == pval_g else None,
        'n': int(model.nobs)
    }

import json
print('\nJSON_OUT')
print(json.dumps(out, indent=2))
