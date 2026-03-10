import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem','min','gender_mf','category','alldeaths','ndam','elapsedyrs','masfem_mturk','wind','ndam15','year']
for c in num_cols:
    _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Drop rows with missing critical values
_df = _df.dropna(subset=['alldeaths','masfem','wind','min','category','year'])

# Transform outcome
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Standardize severity measures for interaction
_df['wind_z'] = (_df['wind'] - _df['wind'].mean()) / _df['wind'].std(ddof=0)
_df['min_z'] = (_df['min'] - _df['min'].mean()) / _df['min'].std(ddof=0)
_df['category_z'] = (_df['category'] - _df['category'].mean()) / _df['category'].std(ddof=0)

# Helper to fit and print model summaries for key terms
models = {}

# Model 1: simple
models['m1'] = smf.ols('log_deaths ~ masfem', data=_df).fit(cov_type='HC3')

# Model 2: controls for severity and time
models['m2'] = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=_df).fit(cov_type='HC3')

# Model 3: interaction with severity (wind)
models['m3'] = smf.ols('log_deaths ~ masfem * wind_z + min + category + year', data=_df).fit(cov_type='HC3')

# Model 4: using gender_mf instead of masfem
models['m4'] = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=_df).fit(cov_type='HC3')

# Model 5: interaction gender with wind
models['m5'] = smf.ols('log_deaths ~ gender_mf * wind_z + min + category + year', data=_df).fit(cov_type='HC3')

# Also check masfem_mturk as robustness
_df = _df.dropna(subset=['masfem_mturk'])
models['m6'] = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=_df).fit(cov_type='HC3')

# Collect key results
out = []
for name, m in models.items():
    for term in ['masfem', 'gender_mf', 'masfem:wind_z', 'gender_mf:wind_z', 'masfem_mturk']:
        if term in m.params.index:
            out.append({
                'model': name,
                'term': term,
                'coef': float(m.params[term]),
                'se': float(m.bse[term]),
                'p': float(m.pvalues[term]),
                'n': int(m.nobs),
                'r2': float(m.rsquared)
            })

# Save results to a csv for inspection
pd.DataFrame(out).to_csv('analysis_results.csv', index=False)

# Print brief results
print(pd.DataFrame(out).sort_values(['model','term']).to_string(index=False))
