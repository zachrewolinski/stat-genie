import json
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import numpy as np

with open('info.json','r') as f:
    info=json.load(f)
fields=info['data_desc']['fields']

# identify gender and approval columns
female_col=None
approval_col=None
denial_col=None
pmi_col=None
for field in fields:
    desc=(field.get('properties',{}).get('description') or '').lower()
    if 'female' in desc:
        female_col=field['column']
    if 'accepted' in desc and 'denied' in desc and '1 if' in desc:
        idx_acc=desc.find('accepted'); idx_den=desc.find('denied')
        if idx_acc!=-1 and idx_den!=-1:
            if idx_acc < idx_den:
                approval_col=field['column']
            else:
                denial_col=field['column']
    if 'private mortgage insurance' in desc:
        pmi_col=field['column']

print('female_col', female_col, 'approval_col', approval_col, 'denial_col', denial_col, 'pmi_col', pmi_col)

df=pd.read_csv('mortgage.csv')

female=df[female_col]
approve=df[approval_col]

mask=female.notna() & approve.notna()
female=female[mask]
approve=approve[mask]

# rates
rate_by_gender=approve.groupby(female).mean().to_dict()
print('approval rates', rate_by_gender)

# contingency and chi-square
ct=pd.crosstab(female, approve)
chi2, p, dof, exp = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# bivariate logit
X=sm.add_constant(female)
model=sm.Logit(approve, X).fit(disp=False)
print('bivariate coef', model.params['denied_PMI' if female_col=='denied_PMI' else female_col], 'p', model.pvalues['denied_PMI' if female_col=='denied_PMI' else female_col])

# adjusted logit (exclude outcome and denial)
controls=[c for c in df.columns if c not in {female_col, approval_col, denial_col}]
X_full=df[[female_col]+controls]
X_full=sm.add_constant(X_full)
mask_full=X_full.notna().all(axis=1) & df[approval_col].notna()
X_full=X_full[mask_full]
y_full=df.loc[mask_full, approval_col]
model_full=sm.Logit(y_full, X_full).fit(disp=False, maxiter=200)
coef=model_full.params[female_col]
pval=model_full.pvalues[female_col]
print('adjusted coef', coef, 'p', pval, 'odds_ratio', float(np.exp(coef)))

