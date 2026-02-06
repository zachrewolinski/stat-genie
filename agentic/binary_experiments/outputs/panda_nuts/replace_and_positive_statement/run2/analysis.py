import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
Df = pd.read_csv('panda_nuts.csv')
Df['sex'] = Df['sex'].astype('category')
Df['help'] = Df['help'].astype('category')

# Efficiency as rate (nuts per second)
Df['rate'] = Df['nuts_opened'] / Df['seconds']
Df['log_rate'] = np.log(Df['rate'] + 1e-6)  # avoid log(0)

# Model 1: Poisson rate model with exposure (seconds)
poisson_model = smf.glm(
    'nuts_opened ~ age + sex + help',
    data=Df,
    family=sm.families.Poisson(),
    offset=np.log(Df['seconds'])
)
poisson_res = poisson_model.fit(cov_type='HC3')

# Model 2: OLS on log rate (robust SEs)
ols_model = smf.ols('log_rate ~ age + sex + help', data=Df)
ols_res = ols_model.fit(cov_type='HC3')

# Sensitivity: exclude extremely short sessions (<5s)
filtered = Df[Df['seconds'] >= 5].copy()
poisson_model_f = smf.glm(
    'nuts_opened ~ age + sex + help',
    data=filtered,
    family=sm.families.Poisson(),
    offset=np.log(filtered['seconds'])
)
poisson_res_f = poisson_model_f.fit(cov_type='HC3')

# Summaries
print('Rows:', len(Df))
print('Rate summary (nuts/sec):')
print(Df['rate'].describe())
print('\nPoisson rate model (all data) coef/pvalues:')
print(pd.DataFrame({'coef': poisson_res.params, 'pvalue': poisson_res.pvalues}))
print('\nOLS log-rate model coef/pvalues:')
print(pd.DataFrame({'coef': ols_res.params, 'pvalue': ols_res.pvalues}))
print('\nPoisson rate model (seconds >= 5) coef/pvalues:')
print(pd.DataFrame({'coef': poisson_res_f.params, 'pvalue': poisson_res_f.pvalues}))
