import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')
print('columns', df.columns.tolist())
print('shape', df.shape)
print(df.head())
print('\nmissing per column')
print(df.isna().sum())

# Basic correlation
if 'beauty' in df.columns and 'eval' in df.columns:
    corr = df[['beauty','eval']].corr().iloc[0,1]
    print('\nPearson correlation beauty-eval:', corr)

# Simple OLS
if {'beauty','eval'}.issubset(df.columns):
    model_simple = smf.ols('eval ~ beauty', data=df).fit()
    print('\nSimple OLS eval ~ beauty')
    print(model_simple.summary())

# Identify possible controls present in dataset
candidate_controls = ['age','gender','minority','native','tenure','students','allstudents','credits','division']
controls = [c for c in candidate_controls if c in df.columns]
print('\ncontrols present', controls)

if {'beauty','eval'}.issubset(df.columns):
    if controls:
        # build formula with categorical handling
        # use C() for categorical columns if dtype object
        terms = ['beauty']
        for c in controls:
            if df[c].dtype == 'object' or df[c].dtype.name == 'category':
                terms.append(f'C({c})')
            else:
                terms.append(c)
        formula = 'eval ~ ' + ' + '.join(terms)
        print('\nformula', formula)
        model = smf.ols(formula, data=df).fit()
        print('\nOLS with controls')
        print(model.summary())

