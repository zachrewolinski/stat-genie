import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('mortgage.csv')
print('shape', df.shape)
print(df.head())
print(df.describe(include='all').T.head(20))

# Check binary columns
binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1})]
print('binary cols', binary_cols)

# Check accept/deny relationship
if 'accept' in df.columns and 'deny' in df.columns:
    print('accept value counts', df['accept'].value_counts())
    print('deny value counts', df['deny'].value_counts())
    print('accept+deny unique', (df['accept'] + df['deny']).unique())

# Correlations with female
if 'female' in df.columns:
    for target in ['accept','deny']:
        if target in df.columns:
            corr = np.corrcoef(df['female'], df[target])[0,1]
            print('corr female vs', target, corr)
            # contingency and chi-square
            ct = pd.crosstab(df['female'], df[target])
            print('crosstab female vs', target)
            print(ct)
            chi2, p, dof, ex = stats.chi2_contingency(ct)
            print('chi2', chi2, 'p', p)

# logistic regression with controls? maybe statsmodels
try:
    import statsmodels.api as sm
    # choose deny as outcome if binary
    if 'deny' in df.columns:
        y = df['deny']
        # Use female plus some controls (all numeric columns except outcomes)
        X_cols = [c for c in df.columns if c not in ['deny','accept']]
        X = df[X_cols].copy()
        X = sm.add_constant(X)
        model = sm.Logit(y, X, missing='drop').fit(disp=False)
        print(model.summary2().tables[1].loc[['female']])
    elif 'accept' in df.columns:
        y = df['accept']
        X_cols = [c for c in df.columns if c not in ['deny','accept']]
        X = df[X_cols].copy()
        X = sm.add_constant(X)
        model = sm.Logit(y, X, missing='drop').fit(disp=False)
        print(model.summary2().tables[1].loc[['female']])
except Exception as e:
    print('logit error', e)
