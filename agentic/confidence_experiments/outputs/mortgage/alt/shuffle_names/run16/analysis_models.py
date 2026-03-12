import json
import pandas as pd
import statsmodels.api as sm

with open('info.json','r') as f:
    info=json.load(f)
fields=info['data_desc']['fields']

# identify gender and approval columns
female_col=None
approval_col=None
for field in fields:
    desc=(field.get('properties',{}).get('description') or '').lower()
    if 'female' in desc:
        female_col=field['column']
    if 'accepted' in desc and 'denied' in desc and '1 if' in desc:
        idx_acc=desc.find('accepted'); idx_den=desc.find('denied')
        if idx_acc!=-1 and idx_den!=-1 and idx_acc < idx_den:
            approval_col=field['column']

print('female_col', female_col, 'approval_col', approval_col)

df=pd.read_csv('mortgage.csv')

y = df[approval_col]

# define control sets
controls_full = [c for c in df.columns if c not in {female_col, approval_col}]
# remove denial outcome if present (1 - approval)
# find denial column by description
for field in fields:
    desc=(field.get('properties',{}).get('description') or '').lower()
    if 'denied' in desc and 'accepted' in desc and '1 if' in desc:
        idx_acc=desc.find('accepted'); idx_den=desc.find('denied')
        if idx_acc>idx_den:
            denial_col=field['column']
            if denial_col in controls_full:
                controls_full.remove(denial_col)
            break

# remove PMI denial control for a reduced model if present
pmi_col=None
for field in fields:
    desc=(field.get('properties',{}).get('description') or '').lower()
    if 'private mortgage insurance' in desc:
        pmi_col=field['column']
        break

controls_reduced = [c for c in controls_full if c != pmi_col]

# helper to fit model

def fit_logit(y, X, label):
    X = sm.add_constant(X)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y2 = y[mask]
    model = sm.Logit(y2, X).fit(disp=False, maxiter=200)
    coef = model.params[female_col]
    p = model.pvalues[female_col]
    print(label, 'coef', coef, 'p', p, 'n', mask.sum())

# model A: female only
fit_logit(y, df[[female_col]], 'bivariate')

# model B: reduced controls (exclude PMI denial)
fit_logit(y, df[[female_col] + controls_reduced], 'controls_no_PMI')

# model C: full controls
fit_logit(y, df[[female_col] + controls_full], 'controls_full')

