import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print('\nunique counts:')
for col in df.columns:
    print(col, df[col].nunique())

# identify binary columns
binary_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
print('binary_cols', binary_cols)

# summarize binary means
for c in binary_cols:
    print(c, df[c].mean())

# check correlations between female and deny/accept
if 'female' in df.columns:
    for target in ['deny','accept','denied_PMI']:
        if target in df.columns:
            ct = pd.crosstab(df['female'], df[target])
            print('crosstab female vs', target)
            print(ct)
            # chi-square
            chi2, p, dof, exp = stats.chi2_contingency(ct)
            print('chi2 p', p)

# logistic regression for likely outcome
# try with deny if binary
if 'female' in df.columns:
    for target in ['deny','accept']:
        if target in df.columns:
            y = df[target]
            if set(y.dropna().unique())<= {0,1}:
                X = sm.add_constant(df[['female']])
                model = sm.Logit(y, X, missing='drop')
                res = model.fit(disp=False)
                print('logit', target)
                print(res.summary())

