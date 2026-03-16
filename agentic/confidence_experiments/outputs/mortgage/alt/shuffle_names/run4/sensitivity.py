import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

with open('info.json','r') as f:
    info=json.load(f)

fields=info['data_desc']['fields']

# find gender column
for field in fields:
    desc=(field.get('properties',{}) or {}).get('description','').lower()
    if 'female' in desc:
        gender_col=field['column']
        break

# find approval/denial columns
approval_col = denial_col = None
for field in fields:
    desc=(field.get('properties',{}) or {}).get('description','').lower()
    if '1 if' not in desc:
        continue
    after = desc.split('1 if',1)[1]
    pos_accept = after.find('accepted')
    pos_deny = after.find('denied')
    if pos_accept != -1 and (pos_deny == -1 or pos_accept < pos_deny):
        approval_col = field['column']
    elif pos_deny != -1 and (pos_accept == -1 or pos_deny < pos_accept):
        denial_col = field['column']

if approval_col is None and denial_col is None:
    raise RuntimeError('could not find outcome column')

_df=pd.read_csv('mortgage.csv')

if approval_col is not None:
    approval = _df[approval_col]
else:
    approval = 1 - _df[denial_col]

female = _df[gender_col]

# build covariates list
n=len(_df)
exclude={approval_col, gender_col}
if approval_col is None and denial_col is not None:
    exclude.add(denial_col)

covariates=[]
for c in _df.columns:
    if c in exclude:
        continue
    if _df[c].nunique(dropna=True) >= n*0.98:
        continue
    covariates.append(c)

# remove perfect predictors
filtered=[]
for c in covariates:
    tmp=pd.concat([approval, _df[c]], axis=1).dropna()
    if tmp.empty:
        continue
    corr = tmp.iloc[:,0].corr(tmp.iloc[:,1])
    if corr is not None and abs(corr)>0.999:
        continue
    if _df[c].dropna().isin([0,1]).all():
        paired = tmp.iloc[:,0] + tmp.iloc[:,1]
        if np.allclose(paired,1):
            continue
    filtered.append(c)

covariates=filtered

# helper to fit logit

def fit_logit(cols, label):
    model_df = pd.concat([approval, female, _df[cols]], axis=1).dropna()
    y = model_df[approval.name]
    X = model_df[cols].copy()
    X[gender_col]=model_df[gender_col]
    X = sm.add_constant(X, has_constant='add')
    try:
        res = sm.Logit(y, X).fit(disp=False)
    except Exception:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    coef = res.params[gender_col]
    pval = res.pvalues[gender_col]
    or_val = float(np.exp(coef))
    print(label, 'coef', coef, 'p', pval, 'odds_ratio', or_val, 'n', len(model_df))

# unadjusted
fit_logit([], 'unadjusted')

# full
fit_logit(covariates, 'full')

# full without column named "female" if present
cov_no_pmi=[c for c in covariates if c!='female']
fit_logit(cov_no_pmi, 'full_no_female_col')

