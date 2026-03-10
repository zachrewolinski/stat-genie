import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Efficiency
# nuts per second
# avoid division by zero just in case
if (df['seconds'] == 0).any():
    df = df.copy()
    df.loc[df['seconds'] == 0, 'seconds'] = np.nan

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Counts
n_sessions = len(df)
n_chimps = df['chimpanzee'].nunique()

# Group means
mean_overall = df['efficiency'].mean()
mean_by_sex = df.groupby('sex')['efficiency'].mean()
mean_by_help = df.groupby('help')['efficiency'].mean()

# Regression with cluster-robust SEs by chimpanzee
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])

# Extract coefficients and p-values
params = ols_cluster.params
pvalues = ols_cluster.pvalues

# R-squared from OLS (same for clustered SEs)
rsq = ols.rsquared

# Simple correlations
a = df['age'].corr(df['efficiency'])

print('n_sessions', n_sessions)
print('n_chimps', n_chimps)
print('mean_overall', mean_overall)
print('mean_by_sex', mean_by_sex.to_dict())
print('mean_by_help', mean_by_help.to_dict())
print('rsq', rsq)
print('corr_age_eff', a)
print('params', params)
print('pvalues', pvalues)
