import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'category', 'wind', 'min', 'ndam', 'ndam15', 'alldeaths', 'year']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Derived variables
# Log deaths to reduce skew
# add 1 to handle zeros

df['log_deaths'] = np.log1p(df['alldeaths'])

# Compute correlations
corr_pearson = df[['masfem', 'alldeaths']].corr(method='pearson').iloc[0,1]
corr_spearman = df[['masfem', 'alldeaths']].corr(method='spearman').iloc[0,1]

# Pearson and Spearman tests
pearson_r, pearson_p = stats.pearsonr(df['masfem'], df['alldeaths'])
spearman_r, spearman_p = stats.spearmanr(df['masfem'], df['alldeaths'])

# Regression models
models = {}

# 1. Simple OLS on log deaths
models['simple'] = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# 2. Control for storm intensity (category, wind, min pressure)
# Some collinearity possible; include category and wind, exclude min if needed
models['intensity'] = smf.ols('log_deaths ~ masfem + category + wind + min', data=df).fit(cov_type='HC3')

# 3. Control for intensity + damages (ndam15)
models['intensity_damage'] = smf.ols('log_deaths ~ masfem + category + wind + min + ndam15', data=df).fit(cov_type='HC3')

# 4. Control for intensity + year trend (year)
models['intensity_year'] = smf.ols('log_deaths ~ masfem + category + wind + min + year', data=df).fit(cov_type='HC3')

# 5. Use masfem_mturk as alternate name femininity
models['mturk_intensity'] = smf.ols('log_deaths ~ masfem_mturk + category + wind + min + year', data=df).fit(cov_type='HC3')

# 6. Binary gender (female=1)
models['gender_intensity'] = smf.ols('log_deaths ~ gender_mf + category + wind + min + year', data=df).fit(cov_type='HC3')

# Collect results
summary = {
    'n': int(df.shape[0]),
    'pearson_corr_masfem_alldeaths': corr_pearson,
    'spearman_corr_masfem_alldeaths': corr_spearman,
    'pearson_test': {'r': pearson_r, 'p': pearson_p},
    'spearman_test': {'r': spearman_r, 'p': spearman_p},
}

# Extract coefficients for masfem in each model
coef_table = {}
for name, model in models.items():
    if 'masfem' in model.params.index:
        term = 'masfem'
    elif 'masfem_mturk' in model.params.index:
        term = 'masfem_mturk'
    else:
        term = 'gender_mf'
    coef_table[name] = {
        'term': term,
        'coef': float(model.params[term]),
        'se': float(model.bse[term]),
        'p': float(model.pvalues[term]),
        'r2': float(model.rsquared)
    }

# Save result to json-like text
import json
output = {'summary': summary, 'coef_table': coef_table}
with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Print key results for quick inspection
print(json.dumps(output, indent=2))
