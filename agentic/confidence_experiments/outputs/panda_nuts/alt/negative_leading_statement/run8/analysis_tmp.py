import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

print('rows', len(df))
print(df.head())

# Compute efficiency: nuts per second
# Avoid division by zero
if (df['seconds'] <= 0).any():
    print('nonpositive seconds', df.loc[df['seconds'] <= 0, 'seconds'].describe())

df['efficiency'] = df['nuts_opened'] / df['seconds']

print('efficiency summary', df['efficiency'].describe())

# Check missing values
print('missing values', df.isna().sum())

# Basic OLS
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print('OLS summary')
print(ols.summary())

# OLS with cluster-robust SE by chimpanzee
ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
print('OLS cluster summary')
print(ols_cluster.summary())

# MixedLM random intercept for chimpanzee
try:
    md = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False, method='lbfgs')
    print('MixedLM summary')
    print(mdf.summary())
except Exception as e:
    print('MixedLM failed', e)

# Also test log efficiency to reduce skew
# Add small constant to avoid log(0)
df['log_eff'] = np.log(df['efficiency'] + 1e-6)
ols_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit()
print('OLS log summary')
print(ols_log.summary())

# ANOVA for overall effect of categorical variables in OLS
anova = sm.stats.anova_lm(ols, typ=2)
print('ANOVA typ2')
print(anova)
