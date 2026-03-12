import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# basic sanity
# define outcome and key predictor
# use accept as outcome
# female: 1 female, 0 male

# compute basic counts
print('rows', len(df))

# check accept/deny consistency
if 'accept' in df.columns and 'deny' in df.columns:
    mismatch = (df['accept'] + df['deny'] != 1).sum()
    print('accept+deny !=1 count', mismatch)

# drop rows with missing in key columns
key_cols = ['accept', 'female']
df_key = df.dropna(subset=key_cols)

# approval rates by gender
rates = df_key.groupby('female')['accept'].agg(['mean','count'])
print('approval rates by female (0=male,1=female):')
print(rates)

# two-proportion z-test
# counts
male = df_key[df_key['female'] == 0]
female = df_key[df_key['female'] == 1]

n1 = len(male)
n2 = len(female)

x1 = male['accept'].sum()
x2 = female['accept'].sum()

p1 = x1 / n1 if n1 else np.nan
p2 = x2 / n2 if n2 else np.nan

# z-test
p_pool = (x1 + x2) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

z = (p1 - p2) / se if se > 0 else np.nan
pval_z = 2 * (1 - stats.norm.cdf(abs(z)))

print('two-proportion z-test: p1 (male) {:.4f}, p2 (female) {:.4f}, z {:.3f}, p {:.4g}'.format(p1,p2,z,pval_z))

# chi-square test of independence
cont = pd.crosstab(df_key['female'], df_key['accept'])
chi2, p_chi, dof, exp = stats.chi2_contingency(cont)
print('chi-square p', p_chi)

# logistic regression: accept ~ female
X = sm.add_constant(df_key[['female']])
y = df_key['accept']

model = sm.Logit(y, X)
res = model.fit(disp=False)
print('logit female only')
print(res.summary())

# odds ratio and CI
params = res.params
conf = res.conf_int()

or_female = np.exp(params['female'])
ci_female = np.exp(conf.loc['female'])
print('OR female', or_female, 'CI', ci_female.values)

# logistic regression with controls
# choose relevant numeric controls from mortgage dataset
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
controls = [c for c in controls if c in df.columns]

# drop rows with missing
df_ctrl = df.dropna(subset=['accept','female'] + controls)
Xc = sm.add_constant(df_ctrl[['female'] + controls])
yc = df_ctrl['accept']

model_c = sm.Logit(yc, Xc)
res_c = model_c.fit(disp=False)
print('logit with controls')
print(res_c.summary())

or_female_c = np.exp(res_c.params['female'])
ci_female_c = np.exp(res_c.conf_int().loc['female'])
print('OR female (controls)', or_female_c, 'CI', ci_female_c.values)

# average marginal effect for female
mfx = res_c.get_margeff(at='overall', method='dydx')
print(mfx.summary())
