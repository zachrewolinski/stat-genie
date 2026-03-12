import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Compute efficiency
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

print('rows', len(_df))
print(_df[['efficiency','nuts_opened','seconds']].describe())
print(_df[['age','sex','help']].head())

# Encode categorical
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# OLS regression
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit()
print(model.summary())

# robust
robust = model.get_robustcov_results(cov_type='HC3')
print('\nRobust HC3:')
print(robust.summary())

# Also check poisson for counts with log(seconds) offset
# Use GLM Poisson: nuts_opened ~ age + sex + help + offset(log(seconds))
_df['log_seconds'] = np.log(_df['seconds'])
poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=_df,
                        family=sm.families.Poisson(), offset=_df['log_seconds']).fit()
print('\nPoisson GLM:')
print(poisson_model.summary())

# Overdispersion check
pearson_chi2 = ((poisson_model.resid_pearson)**2).sum()
df_resid = poisson_model.df_resid
print('Overdispersion ratio (pearson chi2/df):', pearson_chi2/df_resid)

# Negative binomial maybe
nb_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=_df,
                   family=sm.families.NegativeBinomial(), offset=_df['log_seconds']).fit()
print('\nNegBin GLM:')
print(nb_model.summary())

# Compute effect sizes (standardized?) Use standardized coefficients for age? we can compute.
# Standardize age and efficiency for standardized coefficient.
_df['age_z'] = (_df['age'] - _df['age'].mean())/_df['age'].std(ddof=0)
_df['eff_z'] = (_df['efficiency'] - _df['efficiency'].mean())/_df['efficiency'].std(ddof=0)
std_model = smf.ols('eff_z ~ age_z + C(sex) + C(help)', data=_df).fit()
print('\nStandardized model:')
print(std_model.summary())

# Group means
print('\nGroup means:')
print(_df.groupby(['sex','help']).efficiency.mean())
print(_df.groupby('sex').efficiency.describe())
print(_df.groupby('help').efficiency.describe())

