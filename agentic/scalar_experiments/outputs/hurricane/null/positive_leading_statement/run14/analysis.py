import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns
numeric_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'ndam', 'ndam15']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create log deaths
_df = df.copy()
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Standardize intensity measures to compare
def zscore(s):
    return (s - s.mean()) / s.std()

_df['wind_z'] = zscore(_df['wind'])
_df['min_z'] = zscore(_df['min'])
_df['category_z'] = zscore(_df['category'])

# Summary stats
summary = {
    'n': len(_df),
    'deaths_nonzero': int((_df['alldeaths'] > 0).sum()),
    'deaths_zero': int((_df['alldeaths'] == 0).sum()),
    'masfem_mean': float(_df['masfem'].mean()),
    'masfem_sd': float(_df['masfem'].std()),
}

# Models
results = {}

# 1) Bivariate: log deaths ~ masfem
model1 = smf.ols('log_deaths ~ masfem', data=_df).fit(cov_type='HC3')
results['model1'] = model1

# 2) Controls for intensity (wind, min pressure, category)
model2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=_df).fit(cov_type='HC3')
results['model2'] = model2

# 3) Alternative using masfem_mturk
model3 = smf.ols('log_deaths ~ masfem_mturk + wind + min + category', data=_df).fit(cov_type='HC3')
results['model3'] = model3

# 4) Interaction with intensity (wind) as in literature
model4 = smf.ols('log_deaths ~ masfem * wind + min + category', data=_df).fit(cov_type='HC3')
results['model4'] = model4

# 5) Poisson regression on deaths counts (with log link) with robust SE
# Note: use only non-missing
model5 = smf.glm('alldeaths ~ masfem + wind + min + category', data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')
results['model5'] = model5

# Extract key stats
out = {
    'summary': summary,
    'models': {}
}

for name, model in results.items():
    params = model.params
    bse = model.bse
    pvalues = model.pvalues
    # focus on masfem or masfem_mturk and interaction
    keys = [k for k in params.index if 'masfem' in k]
    out['models'][name] = {
        'nobs': int(model.nobs),
        'r2': float(model.rsquared) if hasattr(model, 'rsquared') else None,
        'aic': float(model.aic),
        'coeffs': {k: float(params[k]) for k in keys},
        'ses': {k: float(bse[k]) for k in keys},
        'pvalues': {k: float(pvalues[k]) for k in keys},
    }

# Save outputs to a simple text for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
