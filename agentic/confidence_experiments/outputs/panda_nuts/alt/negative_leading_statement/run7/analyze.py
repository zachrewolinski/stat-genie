import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Efficiency: nuts opened per second
# Avoid divide by zero (shouldn't be any)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Basic summaries
print('rows', len(df))
print('efficiency summary', df['efficiency'].describe())
print('sex counts', df['sex'].value_counts())
print('help counts', df['help'].value_counts())

# Encode categorical
# Use categorical in formula; sex, help

# OLS with cluster-robust SE by chimpanzee (repeated measures)
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})
print('\nOLS cluster-robust')
print(model.summary())

# Also check alternative: log efficiency (add small epsilon)
# to handle skew
# add 1e-6 to avoid log(0)
df['log_eff'] = np.log(df['efficiency'] + 1e-6)
model_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})
print('\nOLS log_eff cluster-robust')
print(model_log.summary())

# Mixed effects with random intercept for chimpanzee
try:
    mixed = smf.mixedlm('efficiency ~ age + C(sex) + C(help)', data=df, groups=df['chimpanzee']).fit(reml=False, method='lbfgs')
    print('\nMixedLM')
    print(mixed.summary())
except Exception as e:
    print('\nMixedLM failed', e)

# Nonparametric correlation for age vs efficiency
from scipy import stats
rho, pval = stats.spearmanr(df['age'], df['efficiency'])
print('\nSpearman age-efficiency', rho, pval)

# Group means for sex and help
print('\nGroup means by sex')
print(df.groupby('sex')['efficiency'].agg(['mean','median','count']))
print('\nGroup means by help')
print(df.groupby('help')['efficiency'].agg(['mean','median','count']))
