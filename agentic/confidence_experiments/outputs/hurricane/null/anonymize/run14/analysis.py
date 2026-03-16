import pandas as pd
import numpy as np
import statsmodels.api as sm
import pingouin as pg

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns to meaningful names for clarity
col_map = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'femininity',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_elapsed',
    'feature11': 'source',
    'feature12': 'femininity_alt',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = _df.rename(columns=col_map)

# Create log deaths (offset by 1 to handle zeros)
df['log_deaths'] = np.log1p(df['deaths'])

# Controls for storm intensity and time
a_controls = ['max_wind', 'min_pressure', 'category', 'year']

# Drop rows with missing values in relevant columns
used_cols = ['log_deaths', 'femininity', 'femininity_alt', 'female_binary'] + a_controls
model_df = df[used_cols].dropna().copy()

# OLS with femininity index
X1 = sm.add_constant(model_df[['femininity'] + a_controls])
y = model_df['log_deaths']
model1 = sm.OLS(y, X1).fit(cov_type='HC3')

# OLS with binary female indicator
X2 = sm.add_constant(model_df[['female_binary'] + a_controls])
model2 = sm.OLS(y, X2).fit(cov_type='HC3')

# OLS with alternate femininity measure
X3 = sm.add_constant(model_df[['femininity_alt'] + a_controls])
model3 = sm.OLS(y, X3).fit(cov_type='HC3')

# Spearman correlation (nonparametric) between femininity and deaths
spearman = pg.corr(df['femininity'], df['deaths'], method='spearman')

# Partial correlation (controlling for intensity + year)
partial = pg.partial_corr(
    data=model_df,
    x='femininity',
    y='log_deaths',
    covar=a_controls,
    method='spearman'
)

# Partial correlation using alternate femininity measure
partial_alt = pg.partial_corr(
    data=model_df,
    x='femininity_alt',
    y='log_deaths',
    covar=a_controls,
    method='spearman'
)

print('N used (model):', len(model_df))
print('Spearman corr (femininity, deaths):')
print(spearman)
print('\nOLS model with femininity index (log deaths):')
print(model1.summary().tables[1])
print('\nOLS model with female binary (log deaths):')
print(model2.summary().tables[1])
print('\nOLS model with alternate femininity measure (log deaths):')
print(model3.summary().tables[1])
print('\nPartial Spearman (femininity vs log deaths, controlling for intensity+year):')
print(partial)
print('\nPartial Spearman (alt femininity vs log deaths, controlling for intensity+year):')
print(partial_alt)
