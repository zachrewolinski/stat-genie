import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

with open('info.json','r') as f:
    info=json.load(f)
fields=info['data_desc']['fields']

# identify gender/approval columns as in analysis.py
female_col=None
approval_col=None
for field in fields:
    desc=(field.get('properties',{}).get('description') or '').lower()
    if 'female' in desc:
        female_col=field['column']
    if 'accepted' in desc and 'denied' in desc and '1 if' in desc:
        idx_one=desc.find('1 if'); idx_acc=desc.find('accepted'); idx_den=desc.find('denied')
        if idx_one!=-1 and idx_acc!=-1 and idx_den!=-1 and idx_acc < idx_den:
            approval_col=field['column']

print('female_col', female_col, 'approval_col', approval_col)

df=pd.read_csv('mortgage.csv')

# correlations with female
female=df[female_col]
print('\nCorrelation with female (Pearson):')
for col in df.columns:
    if col==female_col: continue
    if df[col].dtype.kind in 'biufc':
        corr = np.corrcoef(female, df[col])[0,1]
        print(col, corr)

# difference in means by gender
print('\nMean by gender:')
for col in df.columns:
    if col==female_col: continue
    if df[col].dtype.kind in 'biufc':
        m0=df.loc[female==0, col].mean()
        m1=df.loc[female==1, col].mean()
        print(col, 'mean0', m0, 'mean1', m1, 'diff', m1-m0)

# logistic regression with subset controls: include all numeric except outcome
controls = [c for c in df.columns if c not in {female_col, approval_col}]
X = df[[female_col] + controls]
X = sm.add_constant(X)
mask = X.notna().all(axis=1) & df[approval_col].notna()
X = X[mask]
y = df.loc[mask, approval_col]
model = sm.Logit(y, X).fit(disp=False, maxiter=200)
print('\nFull model female coef', model.params[female_col], 'p', model.pvalues[female_col])

# simple linear probability model
X_lpm = sm.add_constant(df[[female_col]])
mask = X_lpm.notna().all(axis=1) & df[approval_col].notna()
model_lpm = sm.OLS(df.loc[mask, approval_col], X_lpm[mask]).fit()
print('\nLPM coef', model_lpm.params[female_col], 'p', model_lpm.pvalues[female_col])
