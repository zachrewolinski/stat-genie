import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'mortgage.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())

# basic counts for female
if 'female' in df.columns:
    print('female value counts', df['female'].value_counts(dropna=False))

# choose outcome: deny or accept
for col in ['deny','accept']:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False))

# compute denial rates by gender
if 'deny' in df.columns and 'female' in df.columns:
    grp = df.groupby('female')['deny'].mean()
    print('deny mean by female', grp)

# check if accept is inverse of deny
if 'deny' in df.columns and 'accept' in df.columns:
    diff = (df['accept'] == (1 - df['deny'])).mean()
    print('accept == 1 - deny proportion', diff)

# logistic regression: deny ~ female (unadjusted)
if 'deny' in df.columns and 'female' in df.columns:
    X = sm.add_constant(df[['female']])
    model = sm.Logit(df['deny'], X, missing='drop')
    res = model.fit(disp=False)
    print('logit unadjusted', res.summary())

# logistic regression with controls if columns exist
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
controls = [c for c in controls if c in df.columns]
if 'deny' in df.columns and 'female' in df.columns and controls:
    X = sm.add_constant(df[['female'] + controls])
    model = sm.Logit(df['deny'], X, missing='drop')
    res = model.fit(disp=False, maxiter=200)
    print('logit adjusted', res.summary())

# compute predicted probabilities difference adjusted using marginal effects if available
if 'deny' in df.columns and 'female' in df.columns and controls:
    try:
        marg = res.get_margeff(at='overall')
        print('marginal effects', marg.summary())
    except Exception as e:
        print('marginal effects error', e)
