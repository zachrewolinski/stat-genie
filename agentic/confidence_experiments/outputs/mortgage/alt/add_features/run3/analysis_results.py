import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')

# keep relevant columns
cols = ['female','deny','accept','black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
cols = [c for c in cols if c in df.columns]
df = df[cols]

# drop rows with missing female or deny
base = df.dropna(subset=['female','deny'])

# counts and rates
counts = base.groupby('female')['deny'].agg(['count','sum','mean'])
print('deny by female\n', counts)

# difference in proportions test (unadjusted)
# counts for female=1, female=0
f1 = base[base['female']==1]
f0 = base[base['female']==0]
count = np.array([f1['deny'].sum(), f0['deny'].sum()])
nobs = np.array([f1.shape[0], f0.shape[0]])
stat, pval = proportions_ztest(count, nobs)
rate1 = f1['deny'].mean()
rate0 = f0['deny'].mean()
print('unadjusted rates', rate1, rate0, 'diff', rate1-rate0, 'p', pval)

# unadjusted logit
X = sm.add_constant(base[['female']])
res_unadj = sm.Logit(base['deny'], X).fit(disp=False)
print('unadj coef', res_unadj.params)
print('unadj p', res_unadj.pvalues)
print('unadj OR', np.exp(res_unadj.params))
print('unadj OR CI', np.exp(res_unadj.conf_int()))

# adjusted logit
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
controls = [c for c in controls if c in df.columns]
adj = base.dropna(subset=['female','deny']+controls)
X_adj = sm.add_constant(adj[['female'] + controls])
res_adj = sm.Logit(adj['deny'], X_adj).fit(disp=False, maxiter=200)
print('adj coef', res_adj.params['female'])
print('adj p', res_adj.pvalues['female'])
print('adj OR', np.exp(res_adj.params['female']))
print('adj OR CI', np.exp(res_adj.conf_int().loc['female']))

# marginal effect for female
try:
    marg = res_adj.get_margeff(at='overall')
    me = marg.margeff[0]
    se = marg.margeff_se[0]
    z = me / se
    print('adj marginal effect', me, 'se', se, 'z', z, 'p', marg.pvalues[0])
except Exception as e:
    print('marginal effect error', e)
