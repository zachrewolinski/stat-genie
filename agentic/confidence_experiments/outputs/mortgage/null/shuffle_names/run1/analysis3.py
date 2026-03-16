import pandas as pd
import numpy as np


df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0,1})]
num_cols = [c for c in df.columns if c not in binary_cols]

print('binary', binary_cols)
print('num', num_cols)

# correlations between binary and numeric
for b in binary_cols:
    print('\n', b)
    for n in num_cols:
        corr = df[[b,n]].corr().iloc[0,1]
        print(f'{n}: {corr:.3f}')

# logistic regression using statsmodels to see which outcome is best predicted by ratios
import statsmodels.api as sm

# use numeric predictors that are plausible ratios/credit: mortgage_credit, housing_expense_ratio, Unnamed:0, married, black, bad_history??
# but bad_history is likely ID; exclude by using variance or by checking monotonic? We'll exclude if it's almost sequential.

predictors = ['mortgage_credit','housing_expense_ratio','Unnamed: 0','married','black']
X = df[predictors].copy()
X = sm.add_constant(X, has_constant='add')

for b in binary_cols:
    y = df[b]
    # drop missing rows for y and X
    data = pd.concat([y, X], axis=1).dropna()
    y2 = data[b]
    X2 = data[predictors]
    X2 = sm.add_constant(X2, has_constant='add')
    try:
        model = sm.Logit(y2, X2).fit(disp=False)
        llf = model.llf
        print('\nOutcome', b, 'n', len(y2), 'LL', llf)
        print(model.params)
    except Exception as e:
        print('Outcome', b, 'failed', e)

