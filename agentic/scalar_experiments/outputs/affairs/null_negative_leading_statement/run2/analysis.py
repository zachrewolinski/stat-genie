import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Basic group stats
for label, g in df.groupby('children'):
    mean_affairs = g['affairs'].mean()
    any_affair = (g['affairs'] > 0).mean()
    print(f"children={label} n={len(g)} mean_affairs={mean_affairs:.3f} any_affair_rate={any_affair:.3f}")

# Difference in means
mean_yes = df.loc[df['children']=='yes','affairs'].mean()
mean_no = df.loc[df['children']=='no','affairs'].mean()
any_yes = (df.loc[df['children']=='yes','affairs']>0).mean()
any_no = (df.loc[df['children']=='no','affairs']>0).mean()
print("diff mean yes-no", mean_yes-mean_no)
print("diff any yes-no", any_yes-any_no)

# OLS with controls
# Encode children yes/no
# Use robust SE
ols = smf.ols('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(cov_type='HC3')
print(ols.summary())

# Log1p OLS
ols_log = smf.ols('np.log1p(affairs) ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(cov_type='HC3')
print(ols_log.summary())

# Logistic regression for any affair

df['any_affair'] = (df['affairs'] > 0).astype(int)
logit = smf.logit('any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(disp=0)
print(logit.summary())

# Marginal effect for children yes vs no (yes is reference? depends)
# Extract coefficient for C(children)[T.yes]
coef = logit.params.get('C(children)[T.yes]', np.nan)
print('logit coef children yes', coef)

