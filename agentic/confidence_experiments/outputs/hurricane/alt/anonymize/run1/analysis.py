import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns for clarity
# According to info.json:
# feature4: femininity index (1-11)
# feature6: binary female (0 male, 1 female)
# feature8: deaths
# feature5: min pressure
# feature7: category
# feature13: max wind speed
# feature9/14: damages (normalized)

# Basic checks
summary = {
    'n': int(len(df)),
    'deaths_zero': int((df['feature8'] == 0).sum()),
    'deaths_mean': float(df['feature8'].mean()),
    'deaths_median': float(df['feature8'].median()),
    'fem_index_mean': float(df['feature4'].mean()),
    'female_share': float(df['feature6'].mean()),
}

# Correlations with deaths (log1p)
df['log_deaths'] = np.log1p(df['feature8'])

corrs = df[['log_deaths','feature4','feature6','feature7','feature5','feature13','feature9','feature14']].corr()

# OLS: log_deaths ~ fem_index + controls
# Controls: category, min pressure, max wind speed, damage (2013 or 2015) maybe multicollinear.
# We'll use category + min pressure + max wind speed + log damage (feature9)
# Use log damage to reduce skew.

# Prepare log damage
for col in ['feature9','feature14']:
    df[f'log_{col}'] = np.log(df[col])

# Model 1: bivariate
m1 = smf.ols('log_deaths ~ feature4', data=df).fit()
# Model 2: with controls
m2 = smf.ols('log_deaths ~ feature4 + feature7 + feature5 + feature13 + log_feature9', data=df).fit()
# Model 3: use female indicator
m3 = smf.ols('log_deaths ~ feature6 + feature7 + feature5 + feature13 + log_feature9', data=df).fit()

# Robust SE (HC3) versions
m2_hc3 = m2.get_robustcov_results(cov_type='HC3')
m3_hc3 = m3.get_robustcov_results(cov_type='HC3')

# Output key results
results = {
    'summary': summary,
    'corrs_log_deaths': {k: float(v) for k, v in corrs['log_deaths'].to_dict().items()},
    'm1': {
        'coef_fem': float(m1.params['feature4']),
        'pval_fem': float(m1.pvalues['feature4']),
        'r2': float(m1.rsquared),
    },
    'm2': {
        'coef_fem': float(m2.params['feature4']),
        'pval_fem': float(m2.pvalues['feature4']),
        'r2': float(m2.rsquared),
    },
    'm2_hc3': {
        'coef_fem': float(m2_hc3.params[1]),  # feature4
        'pval_fem': float(m2_hc3.pvalues[1]),
    },
    'm3': {
        'coef_female': float(m3.params['feature6']),
        'pval_female': float(m3.pvalues['feature6']),
        'r2': float(m3.rsquared),
    },
    'm3_hc3': {
        'coef_female': float(m3_hc3.params[1]),
        'pval_female': float(m3_hc3.pvalues[1]),
    }
}

# Save results to a json file for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
