import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('mortgage.csv')
print(df.head())
print('shape', df.shape)
print('columns', df.columns.tolist())

# basic checks for outcome variables
for col in ['deny','accept']:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).sort_index())

# check relationship between deny and accept
if 'deny' in df.columns and 'accept' in df.columns:
    ctab = pd.crosstab(df['deny'], df['accept'])
    print('deny vs accept crosstab:\n', ctab)
    print('correlation', df['deny'].corr(df['accept']))

# check female distribution
print('female counts', df['female'].value_counts(dropna=False).sort_index())

# compute approval assuming deny=1 is denied
# So approval = 1 - deny
if 'deny' in df.columns:
    df['approved'] = 1 - df['deny']

    # approval rates by female
    rates = df.groupby('female')['approved'].mean()
    print('approval rates by female', rates)
    # difference and chi-square test
    ctab2 = pd.crosstab(df['female'], df['approved'])
    print('female vs approved crosstab:\n', ctab2)
    chi2, p, dof, expected = stats.chi2_contingency(ctab2)
    print('chi2', chi2, 'p', p)

# compute approval assuming deny=1 is approved (per metadata)
if 'deny' in df.columns:
    rates_deny = df.groupby('female')['deny'].mean()
    print('approval rates by female (deny as approval)', rates_deny)

# logistic regression approve ~ female (unadjusted)
if 'deny' in df.columns:
    y = df['approved']
    X = sm.add_constant(df['female'])
    model = sm.Logit(y, X).fit(disp=False)
    print(model.summary())
    print('odds ratio', np.exp(model.params))

# logistic regression for deny ~ female (alternate)
if 'deny' in df.columns:
    y2 = df['deny']
    X2 = sm.add_constant(df['female'])
    model2 = sm.Logit(y2, X2).fit(disp=False)
    print(model2.summary())
    print('odds ratio deny', np.exp(model2.params))

# multivariate logit using all other columns (treating deny as approval per metadata)
if 'deny' in df.columns:
    feature_cols = [c for c in df.columns if c not in ['deny']]
    X_full = sm.add_constant(df[feature_cols])
    try:
        model_full = sm.Logit(df['deny'], X_full).fit(disp=False)
        print(model_full.summary())
        if 'female' in model_full.params.index:
            print('female coef (full model)', model_full.params['female'], 'p', model_full.pvalues['female'])
    except Exception as e:
        print('full model failed', e)
        missing = df[feature_cols].isna().sum()
        print('missing per column (nonzero):', missing[missing > 0])
        df_full = df[feature_cols + ['deny']].dropna()
        try:
            X_full2 = sm.add_constant(df_full[feature_cols])
            model_full2 = sm.Logit(df_full['deny'], X_full2).fit(disp=False)
            print(model_full2.summary())
            if 'female' in model_full2.params.index:
                print('female coef (full model, dropna)', model_full2.params['female'], 'p', model_full2.pvalues['female'])
        except Exception as e2:
            print('full model dropna failed', e2)
