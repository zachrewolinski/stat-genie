import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
Df = pd.read_csv('teachingratings.csv')

print('shape', Df.shape)
print('columns', list(Df.columns))
print(Df.head())
print(Df.dtypes)

# Basic summary
print(Df[['beauty','eval']].describe())

# Correlation
print('corr', Df['beauty'].corr(Df['eval']))

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=Df).fit()
print(model_simple.summary())

# OLS with controls if present
controls = []
for col in ['age','gender','minority','tenure','native','division','credits','students','allstudents']:
    if col in Df.columns:
        controls.append(col)

if controls:
    # Build formula; treat categorical controls as C()
    terms = ['beauty']
    for col in controls:
        if Df[col].dtype == 'object':
            terms.append(f'C({col})')
        else:
            terms.append(col)
    formula = 'eval ~ ' + ' + '.join(terms)
    model_controls = smf.ols(formula, data=Df).fit()
    print('formula', formula)
    print(model_controls.summary())
