import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('hurricane.csv')

# Coerce relevant columns to numeric, keep NaN for blanks
for col in [
    'masfem','gender_mf','alldeaths','wind','min','category','ndam','ndam15','masfem_mturk'
]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create log-transformed outcomes and damages
# add 1 to handle zeros

df['log_deaths'] = np.log1p(df['alldeaths'])

df['log_ndam15'] = np.log1p(df['ndam15'])

df['log_ndam'] = np.log1p(df['ndam'])

print('Rows:', len(df))
print('Deaths missing:', df['alldeaths'].isna().sum())
print('Masfem missing:', df['masfem'].isna().sum())

# Simple correlation
pearson_r, pearson_p = stats.pearsonr(df['masfem'], df['alldeaths'])
spearman_r, spearman_p = stats.spearmanr(df['masfem'], df['alldeaths'])
print('Pearson r (masfem, alldeaths):', pearson_r, 'p=', pearson_p)
print('Spearman r (masfem, alldeaths):', spearman_r, 'p=', spearman_p)

# Regression: log_deaths ~ masfem + wind + min + category + log_ndam15
# Drop rows with missing values in model
model_cols = ['log_deaths','masfem','wind','min','category','log_ndam15']
model_df = df[model_cols].dropna()
X = model_df[['masfem','wind','min','category','log_ndam15']]
X = sm.add_constant(X)
model = sm.OLS(model_df['log_deaths'], X).fit()
print('\nModel 1: log_deaths ~ masfem + wind + min + category + log_ndam15')
print(model.summary())

# Alternative model using gender_mf (binary) instead of masfem
model_cols2 = ['log_deaths','gender_mf','wind','min','category','log_ndam15']
model_df2 = df[model_cols2].dropna()
X2 = model_df2[['gender_mf','wind','min','category','log_ndam15']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(model_df2['log_deaths'], X2).fit()
print('\nModel 2: log_deaths ~ gender_mf + wind + min + category + log_ndam15')
print(model2.summary())

# Another model with masfem and interaction with severity proxy (wind)
model_cols3 = ['log_deaths','masfem','wind','min','category','log_ndam15']
model_df3 = df[model_cols3].dropna().copy()
model_df3['masfem_x_wind'] = model_df3['masfem'] * model_df3['wind']
X3 = model_df3[['masfem','wind','masfem_x_wind','min','category','log_ndam15']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(model_df3['log_deaths'], X3).fit()
print('\nModel 3: log_deaths ~ masfem + wind + masfem*wind + min + category + log_ndam15')
print(model3.summary())
