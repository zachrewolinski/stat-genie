import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('mortgage.csv')
print('shape', df.shape)
print(df.head())

# identify binary columns (0/1)
binary_cols = []
for col in df.columns:
    vals = df[col].dropna().unique()
    if len(vals) <= 3 and set(vals).issubset({0,1}):
        binary_cols.append(col)
print('binary cols', binary_cols)

# check relationship between deny and accept if present
for a in ['deny','accept']:
    if a in df.columns:
        print(a, df[a].value_counts().to_dict())

if 'deny' in df.columns and 'accept' in df.columns:
    # check if accept is complement of deny
    comp = (df['accept'] + df['deny']).value_counts().to_dict()
    print('accept+deny counts', comp)
    print('accept==1-deny', (df['accept'] == (1-df['deny'])).mean())

# Basic group comparison for female vs deny
if 'female' in df.columns and 'deny' in df.columns:
    grp = df.groupby('female')['deny'].mean()
    cnt = df.groupby('female')['deny'].count()
    print('deny rate by female', grp.to_dict())
    print('counts by female', cnt.to_dict())

# t-test / chi-square for difference in deny rates
if 'female' in df.columns and 'deny' in df.columns:
    # chi-square test
    ct = pd.crosstab(df['female'], df['deny'])
    print('crosstab female x deny')
    print(ct)
    # chi-square
    from scipy.stats import chi2_contingency
    chi2, p, dof, exp = chi2_contingency(ct)
    print('chi2', chi2, 'p', p)

# logistic regression with covariates
# choose covariates: exclude outcome and female
outcome = 'deny' if 'deny' in df.columns else None
if outcome:
    # use numeric columns excluding outcome
    covariates = [c for c in df.columns if c != outcome]
    # drop non-numeric? all numeric
    X = df[covariates].copy()
    # Add intercept
    X = sm.add_constant(X, has_constant='add')
    y = df[outcome]
    # Fit logit
    try:
        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary())
        if 'female' in covariates:
            print('female coef', model.params['female'], 'p', model.pvalues['female'])
    except Exception as e:
        print('logit error', e)
