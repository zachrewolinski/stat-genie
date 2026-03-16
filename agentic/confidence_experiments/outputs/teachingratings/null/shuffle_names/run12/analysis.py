import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data

df = pd.read_csv('teachingratings.csv')

print('Rows', len(df))
print('Columns', df.columns.tolist())

# Basic stats for key vars
print('\nallstudents summary')
print(df['allstudents'].describe())
print('\nbeauty summary')
print(df['beauty'].describe())

# Correlation
corr = df['beauty'].corr(df['allstudents'])
print('\nPearson corr beauty vs allstudents:', corr)

# Simple OLS
X = sm.add_constant(df['beauty'])
model_simple = sm.OLS(df['allstudents'], X).fit()
print('\nSimple OLS: allstudents ~ beauty')
print(model_simple.summary())

# Identify categorical columns (object)
cat_cols = [c for c in df.columns if df[c].dtype == 'object']
print('\nCategorical columns:', cat_cols)
for c in cat_cols:
    print(c, df[c].unique())

# Prepare multivariate regression with controls
# Use all columns except outcome (allstudents)
# Drop identifier-like columns with too many unique values
# We'll include numeric columns (age, beauty, rownames, minority, students?) and dummy-code categorical

outcome = 'allstudents'

# drop potential identifiers: division (unique), students (id-like 1-94) maybe
# We'll try two models: with and without these id-like columns

# Create design matrix
control_cols = [c for c in df.columns if c != outcome]

# model A: exclude division and students (likely identifiers)
exclude_cols = {'division', 'students'}
control_cols_A = [c for c in control_cols if c not in exclude_cols]

X_A = df[control_cols_A].copy()

# dummy code categorical
X_A = pd.get_dummies(X_A, drop_first=True)
X_A = sm.add_constant(X_A)
model_A = sm.OLS(df[outcome], X_A).fit()
print('\nMultivariate OLS (excluding division, students)')
print(model_A.summary())

# model B: include division, students as numeric
control_cols_B = control_cols
X_B = df[control_cols_B].copy()
X_B = pd.get_dummies(X_B, drop_first=True)
X_B = sm.add_constant(X_B)
model_B = sm.OLS(df[outcome], X_B).fit()
print('\nMultivariate OLS (including division, students)')
print(model_B.summary())

# Extract beauty coefficients and p-values

def summarize(model, name):
    if 'beauty' in model.params.index:
        coef = model.params['beauty']
        pval = model.pvalues['beauty']
        se = model.bse['beauty']
        print(f"{name}: beauty coef={coef:.4f}, se={se:.4f}, p={pval:.4g}")

summarize(model_simple, 'Simple')
summarize(model_A, 'Model A')
summarize(model_B, 'Model B')
