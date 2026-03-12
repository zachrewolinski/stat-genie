import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

path = 'teachingratings.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# Basic correlation and regression between beauty and allstudents
# Ensure numeric
beauty = pd.to_numeric(df['beauty'], errors='coerce')
allstudents = pd.to_numeric(df['allstudents'], errors='coerce')
mask = beauty.notna() & allstudents.notna()
print('n pair', mask.sum())

corr = stats.pearsonr(beauty[mask], allstudents[mask])
print('pearson r', corr)

# simple linear regression
X = sm.add_constant(beauty[mask])
model = sm.OLS(allstudents[mask], X).fit()
print(model.summary())

# multiple regression with other numeric variables
# select numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# remove outcome
if 'allstudents' in num_cols:
    num_cols.remove('allstudents')
# include beauty plus other numeric controls
X_multi = df[num_cols].copy()
X_multi = X_multi.apply(pd.to_numeric, errors='coerce')
X_multi = sm.add_constant(X_multi)
mask_multi = X_multi.notna().all(axis=1) & allstudents.notna()
model_multi = sm.OLS(allstudents[mask_multi], X_multi[mask_multi]).fit()
print('numeric columns', num_cols)
print(model_multi.summary())

