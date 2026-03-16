import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure numeric columns are numeric
numeric_cols = ['masfem','masfem_mturk','gender_mf','alldeaths','wind','min','category','ndam','ndam15','year']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create log deaths
if 'alldeaths' in df.columns:
    df['log_deaths'] = np.log1p(df['alldeaths'])

# Drop rows with missing key variables
key_vars = ['alldeaths','masfem','wind','min','category']
model_df = df.dropna(subset=[c for c in key_vars if c in df.columns]).copy()

print('Rows total:', len(df))
print('Rows in model_df:', len(model_df))

# Correlations
if 'alldeaths' in df.columns and 'masfem' in df.columns:
    corr = df[['alldeaths','masfem']].corr().iloc[0,1]
    print('Correlation alldeaths vs masfem:', corr)

# Simple OLS: log_deaths ~ masfem
if 'log_deaths' in df.columns and 'masfem' in df.columns:
    m1 = smf.ols('log_deaths ~ masfem', data=model_df).fit(cov_type='HC3')
    print('\nModel 1: log_deaths ~ masfem (HC3)')
    print(m1.summary().tables[1])

# OLS with controls (wind, min, category, year)
controls = []
for c in ['wind','min','category','year']:
    if c in model_df.columns:
        controls.append(c)

if controls:
    formula = 'log_deaths ~ masfem + ' + ' + '.join(controls)
    m2 = smf.ols(formula, data=model_df).fit(cov_type='HC3')
    print('\nModel 2:', formula, '(HC3)')
    print(m2.summary().tables[1])

# Alternative control using ndam15 instead of min/wind maybe
if all(c in model_df.columns for c in ['ndam15']):
    formula3 = 'log_deaths ~ masfem + wind + min + category + ndam15 + year'
    # drop rows missing ndam15
    m3_df = model_df.dropna(subset=['ndam15'])
    if len(m3_df) > 0:
        m3 = smf.ols(formula3, data=m3_df).fit(cov_type='HC3')
        print('\nModel 3:', formula3, '(HC3)')
        print(m3.summary().tables[1])

# Gender_mf analysis
if 'gender_mf' in model_df.columns:
    m4 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=model_df).fit(cov_type='HC3')
    print('\nModel 4: log_deaths ~ gender_mf + wind + min + category + year (HC3)')
    print(m4.summary().tables[1])

# Interaction with intensity (wind) to test severe effect
if all(c in model_df.columns for c in ['masfem','wind']):
    m5 = smf.ols('log_deaths ~ masfem * wind + min + category + year', data=model_df).fit(cov_type='HC3')
    print('\nModel 5: log_deaths ~ masfem * wind + min + category + year (HC3)')
    print(m5.summary().tables[1])

