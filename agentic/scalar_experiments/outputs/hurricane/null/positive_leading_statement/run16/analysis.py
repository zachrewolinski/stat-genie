import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# Load data

df = pd.read_csv('hurricane.csv')

# Basic prep
# Some columns have names that could conflict; ensure numeric types
num_cols = ['masfem','masfem_mturk','alldeaths','wind','min','category','year','elapsedyrs','ndam','ndam15','gender_mf']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# log transform deaths and damages

df['log_deaths'] = np.log1p(df['alldeaths'])

# Simple correlations
corr_spearman = stats.spearmanr(df['masfem'], df['alldeaths'], nan_policy='omit')
corr_pearson = stats.pearsonr(df['masfem'], df['log_deaths'])

# OLS helper

def ols_summary(y, X, add_const=True):
    if add_const:
        X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    res = model.fit(cov_type='HC3')
    return res

# Models
X1 = df[['masfem']]
res1 = ols_summary(df['log_deaths'], X1)

X2 = df[['masfem','wind','min','category','year']]
res2 = ols_summary(df['log_deaths'], X2)

X3 = df[['gender_mf','wind','min','category','year']]
res3 = ols_summary(df['log_deaths'], X3)

# Negative binomial (count) with controls
X_nb = df[['masfem','wind','min','category','year']]
X_nb = sm.add_constant(X_nb)
nb_model = sm.GLM(df['alldeaths'], X_nb, family=sm.families.NegativeBinomial())
nb_res = nb_model.fit()

# Output key stats
print('N rows:', len(df))
print('Spearman masfem vs deaths:', corr_spearman)
print('Pearson masfem vs log_deaths:', corr_pearson)

print('\nOLS log_deaths ~ masfem')
print(res1.summary().tables[1])

print('\nOLS log_deaths ~ masfem + wind + min + category + year')
print(res2.summary().tables[1])

print('\nOLS log_deaths ~ gender_mf + wind + min + category + year')
print(res3.summary().tables[1])

print('\nNegBin deaths ~ masfem + wind + min + category + year')
print(nb_res.summary().tables[1])

