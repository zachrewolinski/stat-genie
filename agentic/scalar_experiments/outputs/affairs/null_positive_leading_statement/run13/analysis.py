import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
df = pd.read_csv('affairs.csv')

# Clean children
df['children'] = df['children'].astype(str).str.lower().str.strip()

# Outcome measures
df['any_affair'] = (df['affairs'] > 0).astype(int)

# Group stats
group_stats = df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    mean_any=('any_affair','mean')
).reset_index()

# Two-sample t-test for mean affairs
yes = df.loc[df['children']=='yes','affairs']
no = df.loc[df['children']=='no','affairs']
t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Two-proportion z-test for any_affair
yes_any = df.loc[df['children']=='yes','any_affair']
no_any = df.loc[df['children']=='no','any_affair']
n1, n2 = yes_any.size, no_any.size
p1, p2 = yes_any.mean(), no_any.mean()
p_pool = (yes_any.sum()+no_any.sum())/(n1+n2)
se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
z = (p1-p2)/se if se > 0 else np.nan
p_z = 2*(1-stats.norm.cdf(abs(z))) if se > 0 else np.nan

# Logistic regression controlling for covariates
# encode children yes=1
df['children_yes'] = (df['children']=='yes').astype(int)
X = df[['children_yes','age','yearsmarried','religiousness','education','occupation','rating']].copy()
# add gender dummy
df['gender'] = df['gender'].astype(str).str.lower().str.strip()
X['male'] = (df['gender']=='male').astype(int)
X = sm.add_constant(X)
y = df['any_affair']

model = sm.Logit(y, X, missing='drop')
try:
    res = model.fit(disp=False)
    coef = res.params['children_yes']
    p_logit = res.pvalues['children_yes']
    # odds ratio
    oratio = np.exp(coef)
except Exception:
    coef = np.nan
    p_logit = np.nan
    oratio = np.nan

# Save key results to csv for inspection
summary = {
    'mean_affairs_yes': yes.mean(),
    'mean_affairs_no': no.mean(),
    'mean_any_yes': p1,
    'mean_any_no': p2,
    't_p': t_p,
    'z_p': p_z,
    'logit_coef': coef,
    'logit_p': p_logit,
    'logit_or': oratio,
    'n_yes': n1,
    'n_no': n2,
}

print(summary)
