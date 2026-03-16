import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')

print('rows', df.shape)
print('columns', df.columns.tolist())

# Basic sanity
print('missing eval', df['eval'].isna().sum())
print('missing beauty', df['beauty'].isna().sum())

# simple correlation
corr = df[['beauty', 'eval']].corr().iloc[0,1]
print('corr beauty-eval', corr)

# simple OLS
model1 = smf.ols('eval ~ beauty', data=df).fit()
print(model1.summary())

# build multivariate model with common controls if available
# Candidate controls from typical dataset (gender, age, minority, native, tenure, division, credits, students, allstudents)
controls = []
for col in ['gender','age','minority','native','tenure','division','credits','students','allstudents']:
    if col in df.columns:
        controls.append(col)

# use C() for categorical controls
terms = ['beauty']
for col in controls:
    if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
        terms.append(f'C({col})')
    else:
        terms.append(col)

formula = 'eval ~ ' + ' + '.join(terms)
print('formula', formula)

model2 = smf.ols(formula, data=df).fit()
print(model2.summary())

# effect size per sd of beauty
beauty_sd = df['beauty'].std()
coef = model2.params['beauty']
print('beauty sd', beauty_sd)
print('coef (model2)', coef)
print('effect per sd', coef * beauty_sd)

# check if beauty remains significant in model2
print('beauty pvalue', model2.pvalues['beauty'])

# Save a small table for reference
summary = pd.DataFrame({
    'coef': model2.params,
    'se': model2.bse,
    'pvalue': model2.pvalues
})
print(summary.loc[['beauty']])

