import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'teachingratings.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', list(df.columns))
print(df.head())

# basic stats
print(df[['beauty','eval']].describe())

# correlation
corr = df['beauty'].corr(df['eval'])
print('corr', corr)

# simple regression
X = sm.add_constant(df['beauty'])
model = sm.OLS(df['eval'], X).fit()
print(model.summary())

# multivariate regression with plausible controls
# choose available columns
controls = []
for col in ['age','gender','minority','native','tenure','division','credits','students','allstudents']:
    if col in df.columns:
        controls.append(col)
print('controls', controls)

# prepare data: get dummies for categorical
Xc = df[['beauty'] + controls].copy()
# identify categorical columns
cat_cols = Xc.select_dtypes(include=['object','category']).columns
Xc = pd.get_dummies(Xc, columns=cat_cols, drop_first=True)
Xc = sm.add_constant(Xc)
model2 = sm.OLS(df['eval'], Xc).fit()
print(model2.summary())

# effect size: 1 sd increase in beauty -> predicted eval change
sd_beauty = df['beauty'].std()
coef = model2.params['beauty']
print('sd_beauty', sd_beauty, 'coef', coef, 'effect_1sd', coef*sd_beauty)
