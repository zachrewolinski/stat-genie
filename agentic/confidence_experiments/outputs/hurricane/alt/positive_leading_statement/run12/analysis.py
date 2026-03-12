import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Prepare variables
# Use log1p for deaths to handle skew and zeros
# Ensure numeric columns
for col in ['alldeaths','masfem','masfem_mturk','wind','min','category','ndam','ndam15','year','elapsedyrs']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['log_deaths'] = np.log1p(df['alldeaths'])

results = {}

# Model 1: bivariate
m1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')
results['m1'] = m1

# Model 2: control for intensity (wind, min, category)
# These are common severity proxies
m2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC3')
results['m2'] = m2

# Model 3: add year trend (safety improvements / reporting changes)
m3 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')
results['m3'] = m3

# Model 4: use alternative gender coding
m4 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit(cov_type='HC3')
results['m4'] = m4

# Model 5: use MTurk ratings
m5 = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=df).fit(cov_type='HC3')
results['m5'] = m5

# Extract key stats
summary = {}
for name, model in results.items():
    # pick coefficient name
    if 'masfem' in model.params.index:
        key = 'masfem'
    elif 'masfem_mturk' in model.params.index:
        key = 'masfem_mturk'
    elif 'gender_mf' in model.params.index:
        key = 'gender_mf'
    else:
        key = None
    if key:
        summary[name] = {
            'coef': float(model.params[key]),
            'se': float(model.bse[key]),
            'p': float(model.pvalues[key]),
            'n': int(model.nobs),
            'r2': float(model.rsquared),
        }

# Save summary for inspection
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Also compute simple correlations
corrs = {
    'masfem_vs_log_deaths': float(df['masfem'].corr(df['log_deaths'])),
    'masfem_mturk_vs_log_deaths': float(df['masfem_mturk'].corr(df['log_deaths'])),
}
with open('analysis_corrs.json', 'w') as f:
    json.dump(corrs, f, indent=2)

print(json.dumps(summary, indent=2))
