import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
info = json.load(open('info.json'))
df = pd.read_csv('mortgage.csv')

# Identify columns based on descriptions
col_desc = {f['column']: f['properties'].get('description', '') for f in info['data_desc']['fields']}

# Gender column: description contains 'female'
_gender_cols = [c for c, d in col_desc.items() if d and 'female' in d.lower()]
# Denial column: description contains 'denied' and 'mortgage application'
_denial_cols = [c for c, d in col_desc.items() if d and 'mortgage application was denied' in d.lower()]
# Acceptance column: description contains 'mortgage application was accepted'
_accept_cols = [c for c, d in col_desc.items() if d and 'mortgage application was accepted' in d.lower()]

print('gender cols', _gender_cols)
print('denial cols', _denial_cols)
print('accept cols', _accept_cols)

# pick first (should be one each)
if not _gender_cols:
    raise SystemExit('No gender column found')
if not _denial_cols:
    raise SystemExit('No denial column found')

gender_col = _gender_cols[0]
denial_col = _denial_cols[0]

# Build approval as 1 - denial (if denial is 0/1)
df = df.copy()

# Basic sanity
print('gender mean', df[gender_col].mean())
print('denial mean', df[denial_col].mean())

# Drop missing in gender/denial
base = df[[gender_col, denial_col]].dropna()

# Bivariate: approval rate by gender
base['approval'] = 1 - base[denial_col]

rate_by_gender = base.groupby(gender_col)['approval'].mean()
count_by_gender = base.groupby(gender_col)['approval'].count()
print('approval rate by gender', rate_by_gender.to_dict())
print('counts by gender', count_by_gender.to_dict())

# Two-proportion z-test (female=1 vs male=0)
if set(base[gender_col].unique()) <= {0,1} and base[gender_col].nunique() == 2:
    g0 = base[base[gender_col]==0]['approval']
    g1 = base[base[gender_col]==1]['approval']
    n0, n1 = len(g0), len(g1)
    p0, p1 = g0.mean(), g1.mean()
    p_pool = (g0.sum() + g1.sum()) / (n0 + n1)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n0 + 1/n1))
    z = (p1 - p0)/se if se>0 else np.nan
    pval = 2*(1-stats.norm.cdf(abs(z))) if se>0 else np.nan
    print('two-proportion z', z, pval)

# Chi-square test of independence
ct = pd.crosstab(base[gender_col], base[denial_col])
chi2, p_chi, dof, exp = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p_chi)
print('crosstab\n', ct)

# Logistic regression: denial ~ gender + key credit variables (from descriptions)
# select a few controls by description keywords
controls = []
for col, desc in col_desc.items():
    dl = desc.lower() if desc else ''
    if any(k in dl for k in ['ratio of loan amount', 'loan to value', 'debt payments to income', 'housing expense', 'mortgage credit score', 'consumer credit score', 'history of bad credit']):
        controls.append(col)

controls = list(dict.fromkeys(controls))
print('controls', controls)

model_cols = [denial_col, gender_col] + controls
model_df = df[model_cols].dropna()

# Ensure numeric
X = model_df[[gender_col] + controls]
X = sm.add_constant(X, has_constant='add')

y = model_df[denial_col]

try:
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)
    print(res.summary())
    # odds ratio for gender
    coef = res.params[gender_col]
    se = res.bse[gender_col]
    p = res.pvalues[gender_col]
    odds = np.exp(coef)
    # 95% CI for odds ratio
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    print('gender coef', coef, 'odds', odds, 'p', p, 'CI', (ci_low, ci_high))
except Exception as e:
    print('logit failed', e)

