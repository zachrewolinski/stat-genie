import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df))
print('columns', df.columns.tolist())
print('\nmissing counts:')
print(df.isna().sum())

# Convert numeric columns (except name/alldeaths/min source)
# Identify non-numeric columns
non_numeric = ['alldeaths', 'min']
for col in df.columns:
    if col in non_numeric:
        continue
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Rename for clarity
# Based on info.json metadata mapping
# wind -> year, year -> max wind speed
# category -> name femininity rating (1-11)
# gender_mf -> Saffir-Simpson category
# name -> total deaths
# ndam15 -> min pressure
# source, elapsedyrs maybe property damage measures

# Create analysis variables
# deaths
if 'name' in df.columns:
    df['deaths'] = df['name']
# femininity ratings
if 'category' in df.columns:
    df['fem_rating'] = df['category']
if 'ind' in df.columns:
    df['fem_mturk'] = df['ind']
if 'masfem_mturk' in df.columns:
    df['female_binary'] = df['masfem_mturk']
# severity controls
if 'year' in df.columns:
    df['max_wind'] = df['year']
if 'ndam15' in df.columns:
    df['min_pressure'] = df['ndam15']
if 'gender_mf' in df.columns:
    df['ss_category'] = df['gender_mf']
if 'wind' in df.columns:
    df['storm_year'] = df['wind']

# Prepare outcomes
for col in ['deaths', 'elapsedyrs', 'source']:
    if col in df.columns:
        df[f'log1p_{col}'] = np.log1p(df[col])

print('\nSummary deaths:')
print(df['deaths'].describe())

# Simple correlations
print('\nCorrelations with deaths:')
for col in ['fem_rating', 'fem_mturk', 'female_binary']:
    if col in df.columns:
        corr = df[[col, 'deaths']].corr().iloc[0,1]
        print(col, corr)

print('\nCorrelations with damages (log1p_elapsedyrs, log1p_source):')
for outcome in ['log1p_elapsedyrs', 'log1p_source']:
    if outcome in df.columns:
        for col in ['fem_rating', 'fem_mturk', 'female_binary']:
            if col in df.columns:
                corr = df[[col, outcome]].corr().iloc[0,1]
                print(outcome, col, corr)

# Group comparisons for binary female vs male
if 'female_binary' in df.columns:
    grp = df.groupby('female_binary')['deaths'].agg(['count','mean','median'])
    print('\nDeaths by female_binary:')
    print(grp)

# Regression models
results = {}

# Model 1: deaths on fem_rating only
if 'fem_rating' in df.columns:
    model1 = smf.ols('log1p_deaths ~ fem_rating', data=df).fit()
    results['model1'] = model1

# Model 2: add severity controls
controls = []
for c in ['max_wind','min_pressure','ss_category','storm_year']:
    if c in df.columns:
        controls.append(c)

if 'fem_rating' in df.columns:
    if controls:
        formula = 'log1p_deaths ~ fem_rating + ' + ' + '.join(controls)
    else:
        formula = 'log1p_deaths ~ fem_rating'
    model2 = smf.ols(formula, data=df).fit()
    results['model2'] = model2

# Alternative: fem_mturk
if 'fem_mturk' in df.columns:
    if controls:
        formula = 'log1p_deaths ~ fem_mturk + ' + ' + '.join(controls)
    else:
        formula = 'log1p_deaths ~ fem_mturk'
    model3 = smf.ols(formula, data=df).fit()
    results['model3'] = model3

# Binary female
if 'female_binary' in df.columns:
    if controls:
        formula = 'log1p_deaths ~ female_binary + ' + ' + '.join(controls)
    else:
        formula = 'log1p_deaths ~ female_binary'
    model4 = smf.ols(formula, data=df).fit()
    results['model4'] = model4

# Damage outcomes using same controls
for outcome in ['log1p_elapsedyrs', 'log1p_source']:
    if outcome in df.columns and 'fem_rating' in df.columns:
        if controls:
            formula = f'{outcome} ~ fem_rating + ' + ' + '.join(controls)
        else:
            formula = f'{outcome} ~ fem_rating'
        res = smf.ols(formula, data=df).fit()
        results[f'damage_{outcome}_fem_rating'] = res
    if outcome in df.columns and 'fem_mturk' in df.columns:
        if controls:
            formula = f'{outcome} ~ fem_mturk + ' + ' + '.join(controls)
        else:
            formula = f'{outcome} ~ fem_mturk'
        res = smf.ols(formula, data=df).fit()
        results[f'damage_{outcome}_fem_mturk'] = res
    if outcome in df.columns and 'female_binary' in df.columns:
        if controls:
            formula = f'{outcome} ~ female_binary + ' + ' + '.join(controls)
        else:
            formula = f'{outcome} ~ female_binary'
        res = smf.ols(formula, data=df).fit()
        results[f'damage_{outcome}_female_binary'] = res

print('\nRegression summaries (coef, p-value)')
for name, res in results.items():
    coef = res.params
    pvals = res.pvalues
    key = None
    if 'fem_rating' in coef.index:
        key = 'fem_rating'
    elif 'fem_mturk' in coef.index:
        key = 'fem_mturk'
    elif 'female_binary' in coef.index:
        key = 'female_binary'
    print('\n', name)
    if key:
        print('coef', coef[key], 'p', pvals[key])
    print('R2', res.rsquared)

# Output key stats for interpretation
print('\nModel details:')
for name, res in results.items():
    print('\n', name)
    print(res.summary().tables[1])
