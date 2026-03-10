import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
file_path = 'hurricane.csv'
df = pd.read_csv(file_path)

# Prepare variables
# Log-transform deaths to reduce skew; add 1 to handle zeros

df['log_deaths'] = np.log1p(df['alldeaths'])

# Standard control variables for hurricane intensity
# wind: max wind speed at landfall
# min: minimum pressure
# category: Saffir-Simpson category
# year: time trend

# Fit models with robust (HC3) standard errors
models = {}

# Bivariate model
models['bivariate'] = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# Controls for intensity and year
models['controls'] = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit(cov_type='HC3')

# Alternative femininity measure (MTurk)
models['mturk_controls'] = smf.ols('log_deaths ~ masfem_mturk + wind + min + category + year', data=df).fit(cov_type='HC3')

# Binary gender (male/female) for reference
models['gender_controls'] = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=df).fit(cov_type='HC3')

# Output key stats

def extract_stats(model, term):
    if term not in model.params:
        return None
    coef = model.params[term]
    se = model.bse[term]
    p = model.pvalues[term]
    ci_low, ci_high = model.conf_int().loc[term].tolist()
    return coef, se, p, ci_low, ci_high

summary = {}
for name, model in models.items():
    if 'masfem_mturk' in model.params:
        term = 'masfem_mturk'
    elif 'gender_mf' in model.params and name == 'gender_controls':
        term = 'gender_mf'
    else:
        term = 'masfem'
    summary[name] = {
        'n': int(model.nobs),
        'r2': float(model.rsquared),
        'term': term,
        'stats': extract_stats(model, term)
    }

print('SUMMARY')
for name, info in summary.items():
    term = info['term']
    coef, se, p, ci_low, ci_high = info['stats']
    print(f"{name}: term={term}, coef={coef:.4f}, se={se:.4f}, p={p:.4f}, ci=[{ci_low:.4f}, {ci_high:.4f}], r2={info['r2']:.3f}")

# Simple correlations for context
corr_masfem_deaths = df['masfem'].corr(df['log_deaths'])
print(f"corr(masfem, log_deaths)={corr_masfem_deaths:.4f}")

# For practical interpretation: expected percent change per 1 unit masfem in bivariate
coef = summary['bivariate']['stats'][0]
percent_change = (np.exp(coef) - 1) * 100
print(f"bivariate exp(coef)-1 (% change in deaths per 1 unit masfem): {percent_change:.2f}%")
