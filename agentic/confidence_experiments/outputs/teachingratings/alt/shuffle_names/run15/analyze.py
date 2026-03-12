import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
path = 'teachingratings.csv'
df = pd.read_csv(path)

print('shape', df.shape)
print(df.head())
print(df.dtypes)
print('missing', df.isna().sum())

# Identify key columns by name
print('columns', df.columns.tolist())

# Basic stats
print(df.describe(include='all'))

# Try correlation between beauty and allstudents if present
if 'beauty' in df.columns and 'allstudents' in df.columns:
    x = df['beauty']
    y = df['allstudents']
    corr = x.corr(y)
    print('corr beauty vs allstudents', corr)
    # simple OLS
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop').fit()
    print(model.summary())

# For robustness: multiple regression controlling for plausible covariates if exist
covariates = []
for col in ['age','native','gender','credits','tenure','prof','eval','division','students','minority','rownames']:
    if col in df.columns:
        covariates.append(col)

print('covariates', covariates)

# Prepare modeling dataset
if 'beauty' in df.columns and 'allstudents' in df.columns:
    # Create design matrix with numeric encoding for categoricals
    y = df['allstudents']
    X = df[['beauty'] + covariates].copy()
    X = pd.get_dummies(X, drop_first=True)
    X = sm.add_constant(X)
    model2 = sm.OLS(y, X, missing='drop').fit()
    print(model2.summary())

