import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

_df = pd.read_csv('mortgage.csv')

gender_col = 'denied_PMI'  # female indicator per info.json description
approval_col = 'deny'      # acceptance indicator per info.json description

# Drop rows with missing values
_df = _df.dropna().copy()

# approval rate by gender
approval_by_gender = _df.groupby(gender_col)[approval_col].mean()
count_by_gender = _df[gender_col].value_counts().sort_index()

n_f = int(count_by_gender.get(1, 0))
n_m = int(count_by_gender.get(0, 0))
if n_f > 0 and n_m > 0:
    p_f = approval_by_gender.loc[1]
    p_m = approval_by_gender.loc[0]
    pooled = (_df[approval_col].sum()) / len(_df)
    se = np.sqrt(pooled * (1 - pooled) * (1/n_f + 1/n_m))
    z = (p_f - p_m) / se if se > 0 else np.nan
    pval = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
else:
    p_f = p_m = z = pval = np.nan

# Unadjusted odds ratio
if n_f > 0 and n_m > 0:
    approve_f = int(_df.loc[_df[gender_col]==1, approval_col].sum())
    approve_m = int(_df.loc[_df[gender_col]==0, approval_col].sum())
    deny_f = n_f - approve_f
    deny_m = n_m - approve_m
    if min(approve_f, approve_m, deny_f, deny_m) == 0:
        approve_f += 0.5; approve_m += 0.5; deny_f += 0.5; deny_m += 0.5
    or_unadj = (approve_f/deny_f) / (approve_m/deny_m)
else:
    or_unadj = np.nan

# Gender-only logistic regression
X_simple = sm.add_constant(_df[[gender_col]], has_constant='add')
model_simple = sm.GLM(_df[approval_col], X_simple, family=sm.families.Binomial())
res_simple = model_simple.fit()
coef_simple = res_simple.params[gender_col]
se_simple = res_simple.bse[gender_col]
z_simple = coef_simple / se_simple
p_simple = 2 * (1 - stats.norm.cdf(abs(z_simple)))

# Logistic regression with controls
exclude = {approval_col, 'bad_history'}
X_cols = [c for c in _df.columns if c not in exclude]

# Drop columns nearly perfectly correlated with outcome (likely redundant indicators)
corrs = {}
for c in X_cols:
    if _df[c].nunique() > 1:
        corrs[c] = _df[[c, approval_col]].corr().iloc[0,1]
    else:
        corrs[c] = 0.0

high_corr = [c for c, v in corrs.items() if abs(v) > 0.98]
if gender_col in high_corr:
    high_corr.remove(gender_col)
X_cols = [c for c in X_cols if c not in high_corr]

X = _df[X_cols].copy()
X = sm.add_constant(X, has_constant='add')
model = sm.GLM(_df[approval_col], X, family=sm.families.Binomial())
result = model.fit()

coef_gender = result.params.get(gender_col, np.nan)
se_gender = result.bse.get(gender_col, np.nan)
if np.isfinite(coef_gender):
    z_gender = coef_gender / se_gender if se_gender > 0 else np.nan
    p_gender = 2 * (1 - stats.norm.cdf(abs(z_gender))) if np.isfinite(z_gender) else np.nan
    odds_ratio = np.exp(coef_gender)
else:
    z_gender = p_gender = odds_ratio = np.nan

print('n', len(_df))
print('gender counts', count_by_gender.to_dict())
print('approval rates by gender', approval_by_gender.to_dict())
print('diff (female - male)', p_f - p_m)
print('z-test z', z, 'p', pval)
print('unadjusted odds ratio (approval female vs male):', or_unadj)
print('simple logit coef', coef_simple, 'se', se_simple, 'p', p_simple)
print('dropped high-corr columns', high_corr)
print('adjusted logit coef (gender)', coef_gender, 'se', se_gender, 'z', z_gender, 'p', p_gender, 'odds_ratio', odds_ratio)

