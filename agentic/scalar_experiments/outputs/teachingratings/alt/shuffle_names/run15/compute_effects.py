import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('teachingratings.csv')

beauty_sd = df['beauty'].std()
allstudents_sd = df['allstudents'].std()

corr = df['beauty'].corr(df['allstudents'])

X = sm.add_constant(df['beauty'])
model = sm.OLS(df['allstudents'], X).fit()
coef = model.params['beauty']

# With controls
covariates = ['age','native','gender','credits','tenure','prof','eval','division','students','minority','rownames']
X2 = df[['beauty'] + covariates].copy()
X2 = pd.get_dummies(X2, drop_first=True)
X2 = sm.add_constant(X2)
model2 = sm.OLS(df['allstudents'], X2).fit()
coef2 = model2.params['beauty']

print('beauty_sd', beauty_sd)
print('allstudents_sd', allstudents_sd)
print('corr', corr)
print('coef', coef, 'p', model.pvalues['beauty'])
print('coef2', coef2, 'p2', model2.pvalues['beauty'])
print('effect_per_sd_simple', coef * beauty_sd)
print('effect_per_sd_control', coef2 * beauty_sd)

