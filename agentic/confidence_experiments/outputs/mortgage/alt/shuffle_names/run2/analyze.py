import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')
print('columns', df.columns.tolist())
print(df.head())

# identify binary columns (0/1)
# allow float near 0/1
binary_cols = []
for c in df.columns:
    vals = df[c].dropna().unique()
    if len(vals) <= 5:
        # check if all values in {0,1}
        if set(np.round(vals, 6)).issubset({0,1}):
            binary_cols.append(c)
print('binary_cols', binary_cols)

# try to find likely outcome column among deny/accept
for c in df.columns:
    if c in ['deny','accept']:
        print(c, df[c].value_counts().sort_index())

# compute approval rate by female for both possible outcomes
if 'female' in df.columns:
    female = df['female']
    for outcome in ['deny','accept']:
        if outcome in df.columns:
            out = df[outcome]
            # mean outcome by female
            print('mean', outcome, 'female=0/1', df.groupby('female')[outcome].mean())

            # proportion test on outcome=1 by female
            ct = df.groupby('female')[outcome].agg(['sum','count'])
            if 0 in ct.index and 1 in ct.index:
                count = ct['sum'].loc[[0,1]].to_numpy()
                nobs = ct['count'].loc[[0,1]].to_numpy()
                stat, pval = proportions_ztest(count, nobs)
                print('ztest outcome', outcome, 'p=', pval)

# logistic regression: outcome deny ~ female + controls (other columns)

def run_logit(outcome):
    if outcome not in df.columns or 'female' not in df.columns:
        return None
    y = df[outcome]
    # choose controls: all other columns except outcome
    X = df.drop(columns=[outcome])
    # ensure no non-numeric columns
    X = X.apply(pd.to_numeric, errors='coerce')
    # add constant
    X = sm.add_constant(X, has_constant='add')
    # drop rows with missing
    data = pd.concat([y, X], axis=1).dropna()
    y2 = data[outcome]
    X2 = data.drop(columns=[outcome])
    try:
        model = sm.Logit(y2, X2).fit(disp=False)
    except Exception as e:
        print('logit failed', outcome, e)
        return None
    print('logit', outcome, 'n=', len(y2))
    if 'female' in model.params:
        print('female coef', model.params['female'], 'p', model.pvalues['female'])
    return model

run_logit('deny')
run_logit('accept')

