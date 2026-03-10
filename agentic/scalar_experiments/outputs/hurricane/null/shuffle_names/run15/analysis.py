import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print(df.info())
print(df.isna().sum())

# map columns to meanings based on info.json descriptions
# actual meanings:
# wind -> year
# alldeaths -> name
# category -> masfem index (coder ratings)
# ndam15 -> min pressure
# masfem_mturk -> female indicator (0/1)
# gender_mf -> Saffir-Simpson category
# name -> deaths
# elapsedyrs -> damage (normalized 2013)
# masfem -> elapsed years since hurricane
# ind -> MTurk femininity rating (1-11)
# year -> max wind speed
# source -> damage normalized 2015

# Let's rename for analysis clarity
rename = {
    'wind': 'year',
    'alldeaths': 'hurr_name',
    'category': 'masfem_coder',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'ind': 'masfem_mturk',
    'year': 'max_wind',
    'source': 'damage_2015'
}

df2 = df.rename(columns=rename)

# Basic correlations
print('\nCorrelation deaths vs femininity measures:')
for col in ['masfem_coder','masfem_mturk','female_binary']:
    print(col, df2['deaths'].corr(df2[col]))

# log-transform deaths due to skew
# avoid log(0) by adding 1

df2['log_deaths'] = np.log1p(df2['deaths'])

print('\nCorrelation log_deaths vs femininity measures:')
for col in ['masfem_coder','masfem_mturk','female_binary']:
    print(col, df2['log_deaths'].corr(df2[col]))

# Regression: log_deaths ~ femininity + controls (max_wind, min_pressure, ss_category, year)
# controls might include year and damage? but damage is outcome not control.

controls = ['max_wind','min_pressure','ss_category','year']

for fem in ['masfem_coder','masfem_mturk','female_binary']:
    cols = [fem] + controls
    data = df2[cols + ['log_deaths']].dropna()
    X = sm.add_constant(data[cols])
    y = data['log_deaths']
    model = sm.OLS(y, X).fit()
    print(f"\nRegression log_deaths ~ {fem} + controls")
    print(model.summary().tables[1])

# Also model deaths (not log) maybe with negative binomial? but use OLS with robust.

# Print simple mean deaths by female_binary
print('\nMean deaths by female_binary:')
print(df2.groupby('female_binary')['deaths'].mean())
print('Median deaths by female_binary:')
print(df2.groupby('female_binary')['deaths'].median())

# check number of storms
print('Rows:', len(df2))
