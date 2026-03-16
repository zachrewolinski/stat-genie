import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic check
print('rows', len(_df))

# Correlation between beauty and eval
corr = _df['beauty'].corr(_df['eval'])
print('corr', corr)

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=_df).fit()
print(model_simple.summary())

# Multiple regression with controls
# Encode categorical variables with C()
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'
model_ctrl = smf.ols(formula, data=_df).fit()
print(model_ctrl.summary())

# Robust SE (HC3)
model_ctrl_robust = model_ctrl.get_robustcov_results(cov_type='HC3')
print(model_ctrl_robust.summary())

# Effect size: change in eval for 1 SD beauty
beauty_sd = _df['beauty'].std()
coef = model_ctrl.params['beauty']
print('beauty_sd', beauty_sd, 'coef', coef, '1sd_effect', coef*beauty_sd)

# Partial correlation? We'll compute by residualizing eval and beauty on controls
controls = ['age', 'students', 'allstudents']
# add categorical as dummies
cats = ['gender','minority','native','tenure','division','credits']
# Build control design matrix with dummies and intercept
X = pd.get_dummies(_df[controls+cats], drop_first=True)
import statsmodels.api as sm
X = sm.add_constant(X, has_constant='add')
# residualize eval
resid_eval = sm.OLS(_df['eval'], X).fit().resid
resid_beauty = sm.OLS(_df['beauty'], X).fit().resid
partial_corr = np.corrcoef(resid_eval, resid_beauty)[0,1]
# p-value for partial corr using t-test
n = len(_df)
# df = n - k - 2; k = number of controls (including dummies)
# approximate by n - X.shape[1] - 1
k = X.shape[1]-1
# t = r*sqrt(df/(1-r^2))
df = n - k - 2
if df > 0:
    t = partial_corr * np.sqrt(df/(1-partial_corr**2))
    p = 2*stats.t.sf(np.abs(t), df)
else:
    t = np.nan; p = np.nan
print('partial_corr', partial_corr, 't', t, 'p', p, 'df', df)
