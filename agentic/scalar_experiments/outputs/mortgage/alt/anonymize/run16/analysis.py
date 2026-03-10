import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('mortgage.csv')

# Identify columns
female = 'feature2'  # 1 if female, 0 if male
accepted = 'feature14'  # 1 accepted, 0 denied

denied = 'feature11'

# Basic sanity
print('rows', len(df))
print('female unique', df[female].unique())
print('accepted unique', df[accepted].unique())
print('denied unique', df[denied].unique())

# Cross-tab approval rate by gender
# Drop rows with missing gender or outcome for bivariate analysis
df_bi = df.dropna(subset=[female, accepted]).copy()

ct = pd.crosstab(df_bi[female], df_bi[accepted])
print('crosstab\n', ct)

# Approval rates
rates = df_bi.groupby(female)[accepted].mean()
print('approval rates', rates.to_dict())

# Two-proportion z-test / chi-square
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Difference in proportions with CI
n_f = ct.loc[1].sum() if 1 in ct.index else 0
n_m = ct.loc[0].sum() if 0 in ct.index else 0
p_f = ct.loc[1,1] / n_f if n_f>0 else np.nan
p_m = ct.loc[0,1] / n_m if n_m>0 else np.nan

diff = p_f - p_m

# Wald CI for difference in proportions
se = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m)
ci_low = diff - 1.96*se
ci_high = diff + 1.96*se
print('diff', diff, 'CI', (ci_low, ci_high))

# Unadjusted logistic regression: acceptance ~ female
df_unadj = df.dropna(subset=[female, accepted]).copy()
X_unadj = sm.add_constant(df_unadj[[female]], has_constant='add')
model_unadj = sm.Logit(df_unadj[accepted], X_unadj)
res_unadj = model_unadj.fit(disp=False)
coef_u = res_unadj.params[female]
se_u = res_unadj.bse[female]
ci_u = (coef_u - 1.96 * se_u, coef_u + 1.96 * se_u)
or_u = np.exp(coef_u)
ci_or_u = (np.exp(ci_u[0]), np.exp(ci_u[1]))
print('unadjusted female coef', coef_u, 'se', se_u, 'p', res_unadj.pvalues[female])
print('unadjusted odds ratio', or_u, 'CI', ci_or_u)

# Logistic regression: acceptance ~ female + controls
# Use all features except outcome to control.
# drop outcome columns to avoid leakage
features = [c for c in df.columns if c not in [accepted, denied]]

# Drop rows with any missing values in model variables
df_model = df.dropna(subset=features + [accepted]).copy()

X = df_model[features].copy()
X = sm.add_constant(X, has_constant='add')
model = sm.Logit(df_model[accepted], X)
res = model.fit(disp=False)
print(res.summary())

# Extract female coef, odds ratio, CI
coef = res.params[female]
se_f = res.bse[female]
ci = (coef - 1.96*se_f, coef + 1.96*se_f)

odds_ratio = np.exp(coef)
ci_or = (np.exp(ci[0]), np.exp(ci[1]))
print('female coef', coef, 'se', se_f, 'p', res.pvalues[female])
print('odds ratio', odds_ratio, 'CI', ci_or)
