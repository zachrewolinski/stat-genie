import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Clean column names if needed
_df.columns = [c.strip() for c in _df.columns]

# Compute efficiency: nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Basic checks
print('Rows:', len(_df))
print('Missing values:', _df.isna().sum().to_dict())
print('Efficiency summary:', _df['efficiency'].describe())
print('Efficiency zeros:', (_df['efficiency'] == 0).sum())
print('\\nEfficiency by sex (mean, median, n):')
print(_df.groupby('sex')['efficiency'].agg(['mean', 'median', 'count']))
print('\\nEfficiency by help (mean, median, n):')
print(_df.groupby('help')['efficiency'].agg(['mean', 'median', 'count']))

# Encode categorical variables
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# OLS with cluster-robust SE by chimpanzee
formula = 'efficiency ~ age + C(sex) + C(help)'
ols = smf.ols(formula, data=_df).fit()
ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=_df['chimpanzee'])
print('\nOLS (cluster-robust by chimpanzee)')
print(ols_cluster.summary())

# Poisson GLM with offset for time (seconds) to model rate
# Avoid log(0) by ensuring seconds > 0
_df['log_seconds'] = np.log(_df['seconds'])
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=_df,
                  family=sm.families.Poisson(), offset=_df['log_seconds']).fit()
print('\nPoisson GLM with offset')
print(poisson.summary())

# Check overdispersion: Pearson chi2 / df
pearson_chi2 = sum(poisson.resid_pearson**2)
df_resid = poisson.df_resid
print('\nPoisson overdispersion (Pearson chi2 / df):', pearson_chi2 / df_resid)

# Negative binomial GLM (if overdispersion > 1.5, likely)
try:
    nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=_df,
                 family=sm.families.NegativeBinomial(), offset=_df['log_seconds']).fit()
    print('\nNegative Binomial GLM with offset')
    print(nb.summary())
except Exception as e:
    print('Negative Binomial failed:', e)

# GEE with Negative Binomial to account for repeated measures
try:
    gee_nb = sm.GEE.from_formula(
        'nuts_opened ~ age + C(sex) + C(help)',
        groups='chimpanzee',
        data=_df,
        family=sm.families.NegativeBinomial(),
        offset=_df['log_seconds']
    ).fit()
    print('\nGEE Negative Binomial with offset (clustered by chimpanzee)')
    print(gee_nb.summary())
except Exception as e:
    print('GEE Negative Binomial failed:', e)

# Discrete Negative Binomial with estimated alpha, cluster-robust SE
try:
    X = pd.get_dummies(_df[['age', 'sex', 'help']], drop_first=True)
    X = sm.add_constant(X)
    nb2 = sm.NegativeBinomial(_df['nuts_opened'], X, offset=_df['log_seconds']).fit(disp=0)
    nb2_cluster = nb2.get_robustcov_results(cov_type='cluster', groups=_df['chimpanzee'])
    print('\nNegative Binomial (discrete) with estimated alpha, cluster-robust SE')
    print(nb2_cluster.summary())
except Exception as e:
    print('Discrete Negative Binomial failed:', e)

# Effect sizes for OLS (standardized coefficients)
# Standardize continuous age and efficiency for standardized beta
_df_std = _df.copy()
_df_std['age_z'] = (_df_std['age'] - _df_std['age'].mean()) / _df_std['age'].std()
_df_std['efficiency_z'] = (_df_std['efficiency'] - _df_std['efficiency'].mean()) / _df_std['efficiency'].std()
ols_std = smf.ols('efficiency_z ~ age_z + C(sex) + C(help)', data=_df_std).fit()
ols_std_cluster = ols_std.get_robustcov_results(cov_type='cluster', groups=_df_std['chimpanzee'])
print('\nStandardized OLS (cluster-robust)')
print(ols_std_cluster.summary())

# Nonparametric / correlation checks
male = _df.loc[_df['sex'] == 'm', 'efficiency']
female = _df.loc[_df['sex'] == 'f', 'efficiency']
help_yes = _df.loc[_df['help'] == 'y', 'efficiency']
help_no = _df.loc[_df['help'] == 'N', 'efficiency']

print('\nMann-Whitney U: efficiency by sex')
print(stats.mannwhitneyu(male, female, alternative='two-sided'))

print('\nMann-Whitney U: efficiency by help')
print(stats.mannwhitneyu(help_yes, help_no, alternative='two-sided'))

print('\nSpearman correlation: age vs efficiency')
print(stats.spearmanr(_df['age'], _df['efficiency']))
