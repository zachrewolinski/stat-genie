import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Basic cleaning/encoding
# children: yes/no
if 'children' not in df.columns:
    raise ValueError('children column missing')

# normalize to lowercase string
children = df['children'].astype(str).str.strip().str.lower()

df = df.assign(children_yes=(children == 'yes').astype(int))

# affairs numeric
if 'affairs' not in df.columns:
    raise ValueError('affairs column missing')

affairs = pd.to_numeric(df['affairs'], errors='coerce')
df = df.assign(affairs=affairs)

# any affair
any_affair = (df['affairs'] > 0).astype(int)
df = df.assign(any_affair=any_affair)

# Group stats
by_child = df.groupby('children_yes')
mean_affairs = by_child['affairs'].mean()
std_affairs = by_child['affairs'].std()
count_affairs = by_child['affairs'].count()
prop_any = by_child['any_affair'].mean()

# Welch t-test for mean affairs
grp_yes = df.loc[df['children_yes'] == 1, 'affairs']
grp_no = df.loc[df['children_yes'] == 0, 'affairs']

t_stat, t_p = stats.ttest_ind(grp_yes, grp_no, equal_var=False, nan_policy='omit')

# Difference in proportions (z-test)
# compute z for two proportions
n1 = grp_yes.shape[0]
n0 = grp_no.shape[0]

p1 = df.loc[df['children_yes'] == 1, 'any_affair'].mean()
p0 = df.loc[df['children_yes'] == 0, 'any_affair'].mean()

p_pool = (df.loc[df['children_yes'] == 1, 'any_affair'].sum() + df.loc[df['children_yes'] == 0, 'any_affair'].sum()) / (n1 + n0)

se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
if se_pool == 0:
    z_stat = np.nan
    z_p = np.nan
else:
    z_stat = (p1 - p0) / se_pool
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Logistic regression with controls if columns exist
controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
# gender as binary
if 'gender' in df.columns:
    gender = df['gender'].astype(str).str.strip().str.lower()
    df = df.assign(gender_male=(gender == 'male').astype(int))
    controls.append('gender_male')

X_cols = ['children_yes'] + [c for c in controls if c in df.columns]

# prepare model data
model_df = df[X_cols + ['any_affair']].dropna()

X = model_df[X_cols]
X = sm.add_constant(X, has_constant='add')

y = model_df['any_affair']

try:
    logit_model = sm.Logit(y, X).fit(disp=False)
    logit_coef = logit_model.params['children_yes']
    logit_p = logit_model.pvalues['children_yes']
except Exception as e:
    logit_model = None
    logit_coef = np.nan
    logit_p = np.nan

# OLS for affairs count (simple)
ols_df = df[X_cols + ['affairs']].dropna()
X_ols = sm.add_constant(ols_df[X_cols], has_constant='add')
y_ols = ols_df['affairs']

try:
    ols_model = sm.OLS(y_ols, X_ols).fit()
    ols_coef = ols_model.params['children_yes']
    ols_p = ols_model.pvalues['children_yes']
except Exception as e:
    ols_model = None
    ols_coef = np.nan
    ols_p = np.nan

print('Counts (children_yes=1,0):', n1, n0)
print('Mean affairs (yes,no):', mean_affairs.get(1, np.nan), mean_affairs.get(0, np.nan))
print('Std affairs (yes,no):', std_affairs.get(1, np.nan), std_affairs.get(0, np.nan))
print('Prop any affair (yes,no):', p1, p0)
print('Welch t-test: t=%.4f p=%.6f' % (t_stat, t_p))
print('Prop z-test: z=%.4f p=%.6f' % (z_stat, z_p))
print('Logit coef (children_yes):', logit_coef, 'p=', logit_p)
print('OLS coef (children_yes):', ols_coef, 'p=', ols_p)
