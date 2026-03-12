import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'teachingratings.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print('columns', df.columns.tolist())
print(df.dtypes)

# basic correlation
if 'beauty' in df.columns and 'eval' in df.columns:
    corr = df['beauty'].corr(df['eval'])
    print('corr beauty-eval', corr)

# simple OLS
if 'beauty' in df.columns and 'eval' in df.columns:
    model = smf.ols('eval ~ beauty', data=df).fit()
    print(model.summary())

# add common controls if exist
controls = []
for col in ['age','female','gender','minority','tenure','native','division','credits','students','allstudents']:
    if col in df.columns:
        controls.append(col)

print('controls', controls)

# build formula with categorical handling for string/bool
if 'beauty' in df.columns and 'eval' in df.columns:
    # ensure categorical for object columns
    cat_terms = []
    num_terms = []
    for c in controls:
        if df[c].dtype == 'object':
            cat_terms.append(f'C({c})')
        else:
            num_terms.append(c)
    terms = ['beauty'] + num_terms + cat_terms
    formula = 'eval ~ ' + ' + '.join(terms)
    print('formula', formula)
    model2 = smf.ols(formula, data=df).fit()
    print(model2.summary())

