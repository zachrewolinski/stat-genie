import json
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# Load metadata
with open('info.json','r') as f:
    info=json.load(f)

fields=info['data_desc']['fields']

def find_col(keyword):
    for field in fields:
        desc=(field.get('properties',{}) or {}).get('description','')
        if keyword in desc.lower():
            return field['column'], desc
    return None, None

# Identify gender column (female)
gender_col, gender_desc = find_col('female')

def indicator_meaning(desc: str):
    desc_l = desc.lower()
    if "1 if" not in desc_l:
        return None
    after = desc_l.split("1 if", 1)[1]
    pos_accept = after.find("accepted")
    pos_deny = after.find("denied")
    if pos_accept != -1 and (pos_deny == -1 or pos_accept < pos_deny):
        return "accept"
    if pos_deny != -1 and (pos_accept == -1 or pos_deny < pos_accept):
        return "deny"
    return None

# Identify approval/denial columns based on description semantics
approval_col = approval_desc = denial_col = denial_desc = None
for field in fields:
    desc = (field.get('properties', {}) or {}).get('description', '')
    meaning = indicator_meaning(desc)
    if meaning == "accept" and approval_col is None:
        approval_col, approval_desc = field['column'], desc
    elif meaning == "deny" and denial_col is None:
        denial_col, denial_desc = field['column'], desc

print('gender_col', gender_col, gender_desc)
print('approval_col', approval_col, approval_desc)
print('denial_col', denial_col, denial_desc)

# Load data
_df=pd.read_csv('mortgage.csv')

# Build approval indicator
if approval_col is not None:
    approval = _df[approval_col]
elif denial_col is not None:
    approval = 1 - _df[denial_col]
else:
    raise RuntimeError("Could not identify approval/denial column from metadata.")

# Gender indicator
female = _df[gender_col]

# Drop missing
base=pd.DataFrame({'approval': approval, 'female': female})
base=base.dropna()

# Summary rates
rates=base.groupby('female')['approval'].mean()
counts=base.groupby('female')['approval'].agg(['sum','count'])

print('approval rates by female')
print(rates)
print('counts')
print(counts)

# two-proportion z-test (female=1 vs male=0)
if set(counts.index) >= {0,1}:
    successes = np.array([counts.loc[1,'sum'], counts.loc[0,'sum']])
    nobs = np.array([counts.loc[1,'count'], counts.loc[0,'count']])
    zstat, pval = proportions_ztest(successes, nobs)
    diff = rates.loc[1] - rates.loc[0]
    print('diff', diff, 'z', zstat, 'p', pval)
else:
    print('gender groups not complete')

# Build model covariates: all columns except id-like, approval, female
# Identify id-like columns as those with nunique == n or very high unique
n = len(_df)
exclude = {approval_col, gender_col}
if approval_col is None and denial_col is not None:
    exclude.add(denial_col)

# remove exact row index column by high uniqueness
covariates = []
for c in _df.columns:
    if c in exclude:
        continue
    nun = _df[c].nunique(dropna=True)
    if nun >= n * 0.98:
        continue
    covariates.append(c)

# drop any covariate that perfectly predicts approval (correlation ~1 or -1)
filtered_covariates = []
for c in covariates:
    s = _df[c]
    tmp = pd.concat([approval, s], axis=1).dropna()
    if tmp.empty:
        continue
    corr = tmp.iloc[:, 0].corr(tmp.iloc[:, 1])
    if corr is not None and abs(corr) > 0.999:
        continue
    # also drop exact complements for binary columns
    if s.dropna().isin([0,1]).all():
        paired = tmp.iloc[:, 0] + tmp.iloc[:, 1]
        if np.allclose(paired, 1):
            continue
    filtered_covariates.append(c)

covariates = filtered_covariates

X=_df[covariates].copy()
# drop rows with missing in X or y
model_df = pd.concat([approval, female, X], axis=1)
model_df = model_df.dropna()

# rename for convenience
y = model_df[approval.name]
X = model_df[covariates].copy()
# include gender as predictor
X[gender_col] = model_df[gender_col]
X = sm.add_constant(X, has_constant='add')

print('covariates', covariates)

# fit logistic regression
coef = pvalue = None
try:
    model = sm.Logit(y, X)
    res = model.fit(disp=False)
    coef = res.params[gender_col]
    pvalue = res.pvalues[gender_col]
    print('logit coef', coef, 'p', pvalue)
except PerfectSeparationError as e:
    print('perfect separation', e)
except Exception as e:
    print('logit error', e)
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        res = model.fit()
        coef = res.params[gender_col]
        pvalue = res.pvalues[gender_col]
        print('glm coef', coef, 'p', pvalue)
    except Exception as e2:
        print('glm error', e2)

if coef is not None:
    odds_ratio = float(np.exp(coef))
    print('odds_ratio', odds_ratio)
