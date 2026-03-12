import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

# Ensure categorical types
for col in ['sex', 'help']:
    df[col] = df[col].astype('category')

# Efficiency: nuts opened per second
# Avoid division by zero (seconds min 2.5 per metadata)
df['efficiency'] = df['nuts_opened'] / df['seconds']

formula = 'nuts_opened ~ age + C(sex) + C(help)'

# Poisson regression for counts with exposure offset
poisson_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
)
poisson_res = poisson_model.fit(
    cov_type='cluster',
    cov_kwds={'groups': df['chimpanzee']}
)

# Overdispersion check
pearson_chi2 = np.sum(poisson_res.resid_pearson**2)
overdispersion = pearson_chi2 / poisson_res.df_resid

# Negative binomial (NB2) using discrete model to estimate alpha
# Build design matrix
X = sm.add_constant(pd.get_dummies(df[['age', 'sex', 'help']], drop_first=True))
nb2_model = sm.NegativeBinomial(
    endog=df['nuts_opened'],
    exog=X,
    loglike_method='nb2',
    exposure=df['seconds']
)
nb2_res = nb2_model.fit(disp=False)

# Cluster-robust covariance for NB2 via sandwich estimator
nb2_cov = cov_cluster(nb2_res, df['chimpanzee'])
nb2_se = np.sqrt(np.diag(nb2_cov))
nb2_z = nb2_res.params / nb2_se
nb2_p = 2 * (1 - stats.norm.cdf(np.abs(nb2_z)))
nb2_ci_low = nb2_res.params - 1.96 * nb2_se
nb2_ci_high = nb2_res.params + 1.96 * nb2_se

# OLS on efficiency as a secondary check
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df)
ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

print('Rows', len(df))
print('Efficiency summary (nuts/sec):')
print(df['efficiency'].describe())
print('Overdispersion (Poisson Pearson chi2/df):', overdispersion)

print('\nPoisson (cluster-robust) coefficients:')
print(poisson_res.summary().tables[1])

print('\nNegative binomial NB2 (cluster-robust via sandwich) coefficients:')
nb2_table = pd.DataFrame({
    'coef': nb2_res.params,
    'std_err': nb2_se,
    'z': nb2_z,
    'p_value': nb2_p,
    'ci_low': nb2_ci_low,
    'ci_high': nb2_ci_high,
})
print(nb2_table)

print('\nOLS on efficiency (cluster-robust) coefficients:')
print(ols_res.summary().tables[1])

# Rate ratios and 95% CI for Poisson
params = poisson_res.params
conf = poisson_res.conf_int()
rr = np.exp(params)
rr_ci = np.exp(conf)
rate_ratio_table = pd.DataFrame({
    'rate_ratio': rr,
    'ci_low': rr_ci[0],
    'ci_high': rr_ci[1],
    'p_value': poisson_res.pvalues
})
print('\nRate ratios (Poisson, cluster-robust):')
print(rate_ratio_table)

# Rate ratios for NB2
rr_nb2 = np.exp(nb2_res.params)
rr_nb2_ci_low = np.exp(nb2_ci_low)
rr_nb2_ci_high = np.exp(nb2_ci_high)
rate_ratio_table_nb2 = pd.DataFrame({
    'rate_ratio': rr_nb2,
    'ci_low': rr_nb2_ci_low,
    'ci_high': rr_nb2_ci_high,
    'p_value': nb2_p
})
print('\nRate ratios (NB2, cluster-robust via sandwich):')
print(rate_ratio_table_nb2)

print('\nNB2 alpha (dispersion):', nb2_res.params[-1])
