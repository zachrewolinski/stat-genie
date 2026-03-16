import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# Ensure expected columns exist
cols_needed = ['age', 'sex', 'help', 'nuts_opened', 'seconds']
missing = [c for c in cols_needed if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# Drop rows with missing relevant fields
work = df[cols_needed].copy()
work = work.dropna()

# Ensure seconds > 0
work = work[work['seconds'] > 0].copy()

# Efficiency: nuts per second
work['efficiency'] = work['nuts_opened'] / work['seconds']

# Model 1: OLS on efficiency
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=work).fit(cov_type='HC3')

# Model 2: OLS on log efficiency (add small constant to avoid log(0))
work['log_efficiency'] = np.log(work['efficiency'] + 1e-6)
ols_log = smf.ols('log_efficiency ~ age + C(sex) + C(help)', data=work).fit(cov_type='HC3')

# Model 3: Poisson on counts with offset log(seconds)
# Use GLM Poisson with robust SE
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=work,
                  family=sm.families.Poisson(), offset=np.log(work['seconds'])).fit(cov_type='HC3')

# Summary stats by groups
summary = {
    'n': len(work),
    'efficiency_mean': work['efficiency'].mean(),
    'efficiency_std': work['efficiency'].std(),
}

# Group means
summary['by_sex'] = work.groupby('sex')['efficiency'].agg(['mean','std','count']).to_dict('index')
summary['by_help'] = work.groupby('help')['efficiency'].agg(['mean','std','count']).to_dict('index')

# Correlation with age
summary['age_corr'] = work[['age','efficiency']].corr().iloc[0,1]

# ANOVA-like F-test for age+sex+help jointly in OLS
# Compare with intercept-only model
ols_null = smf.ols('efficiency ~ 1', data=work).fit(cov_type='HC3')
# Wald test for all predictors except intercept
# Use ols.wald_test for all coefficients except intercept
param_names = [p for p in ols.params.index if p != 'Intercept']
R = np.eye(len(ols.params))[1:]
wald = ols.wald_test(R)

print('N', summary['n'])
print('Mean efficiency', summary['efficiency_mean'])
print('Std efficiency', summary['efficiency_std'])
print('Age corr', summary['age_corr'])
print('By sex', summary['by_sex'])
print('By help', summary['by_help'])

print('\nOLS efficiency (HC3)')
print(ols.summary())
print('\nOLS log efficiency (HC3)')
print(ols_log.summary())
print('\nPoisson with offset (HC3)')
print(poisson.summary())
print('\nWald test (all predictors)')
print(wald)
