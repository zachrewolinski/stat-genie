import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

_df = pd.read_csv('mortgage.csv')

controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']
cols = ['female', 'accept'] + controls
_df_sub = _df[cols].dropna()

rates = _df_sub.groupby('female')['accept'].mean()
counts = _df_sub.groupby('female')['accept'].agg(['count','sum'])

count = counts['sum'].values
nobs = counts['count'].values
stat, pval = proportions_ztest(count, nobs)

ct = pd.crosstab(_df_sub['female'], _df_sub['accept'])
chi2, chi_p, _, _ = chi2_contingency(ct)

X = _df_sub[['female'] + controls]
X = sm.add_constant(X)

y = _df_sub['accept']

logit_model = sm.Logit(y, X)
res = logit_model.fit(disp=False)

coef = res.params['female']
se = res.bse['female']
p_logit = res.pvalues['female']
odds_ratio = float(np.exp(coef))
ci = res.conf_int().loc['female']
ci_or = np.exp(ci)

marg_eff = res.get_margeff(at='mean', method='dydx')
# Map to variable names
exog_names = res.model.exog_names
# get_margeff usually excludes the constant
if len(marg_eff.margeff) == len(exog_names):
    meff_names = exog_names
else:
    meff_names = [n for n in exog_names if n != 'const']
idx = meff_names.index('female')
meff = float(marg_eff.margeff[idx])
meff_se = float(marg_eff.margeff_se[idx])
meff_p = float(marg_eff.pvalues[idx])

print('Rows used:', len(_df_sub))
print('Approval rates by female (0=male,1=female):')
print(rates)
print('Counts by female:')
print(counts)
print('Two-proportion z-test: z=%.4f p=%.6f' % (stat, pval))
print('Chi-square test: chi2=%.4f p=%.6f' % (chi2, chi_p))
print('Logit female coef=%.4f (se=%.4f) p=%.6f OR=%.4f 95%% CI OR=[%.4f, %.4f]' % (
    coef, se, p_logit, odds_ratio, ci_or[0], ci_or[1]
))
print('Marginal effect of female (dP/dFemale at means)=%.6f (se=%.6f) p=%.6f' % (meff, meff_se, meff_p))

# simple model accept ~ female only
X2 = sm.add_constant(_df_sub[['female']])
logit2 = sm.Logit(y, X2).fit(disp=False)
coef2 = logit2.params['female']
se2 = logit2.bse['female']
p2 = logit2.pvalues['female']
OR2 = np.exp(coef2)
ci2 = np.exp(logit2.conf_int().loc['female'])
print('Logit (female only) coef=%.4f (se=%.4f) p=%.6f OR=%.4f 95%% CI OR=[%.4f, %.4f]' % (
    coef2, se2, p2, OR2, ci2[0], ci2[1]
))

