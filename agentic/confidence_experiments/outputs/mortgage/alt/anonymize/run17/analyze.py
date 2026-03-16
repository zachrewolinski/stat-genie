import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats


df = pd.read_csv('mortgage.csv')

# Variables
female = df['feature2']  # 1 if female
accepted = df['feature14']  # 1 accepted

# Descriptive acceptance rates
rate_tbl = df.groupby(female)['feature14'].agg(['mean','count','sum'])
print('Acceptance by female:\n', rate_tbl)

# Two-proportion z-test (female vs male)
count = rate_tbl['sum'].values
nobs = rate_tbl['count'].values
stat, pval = proportions_ztest(count, nobs)
print('Two-proportion z-test stat', stat, 'p', pval)

# Chi-square test of independence
cont = pd.crosstab(female, accepted)
chi2, pchi, dof, expected = stats.chi2_contingency(cont)
print('Chi-square', chi2, 'p', pchi)

# Logistic regression: accepted on female + controls
controls = ['feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature12','feature13']
model_df = df[['feature2'] + controls + ['feature14']].replace([np.inf, -np.inf], np.nan).dropna()

X = model_df[['feature2'] + controls]
X = sm.add_constant(X)
y = model_df['feature14']

logit = sm.Logit(y, X)
res = logit.fit(disp=False)
print(res.summary())

# Odds ratio for female
params = res.params
conf = res.conf_int()
or_female = float(np.exp(params['feature2']))
ci_low, ci_high = np.exp(conf.loc['feature2'])
print('OR female', or_female, 'CI', (float(ci_low), float(ci_high)), 'p', float(res.pvalues['feature2']))
print('N used', len(model_df))

