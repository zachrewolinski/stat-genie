import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

path = 'hurricane.csv'
df = pd.read_csv(path)

df['log_deaths'] = np.log1p(df['feature8'])

# MTurk femininity rating (feature12)
model_a = smf.ols('log_deaths ~ feature12 + feature7 + feature13 + feature5 + feature2', data=df).fit(cov_type='HC3')
model_b = smf.ols('log_deaths ~ feature12', data=df).fit(cov_type='HC3')

print({
    'model_a_coef': model_a.params['feature12'],
    'model_a_p': model_a.pvalues['feature12'],
    'model_a_ci': model_a.conf_int().loc['feature12'].tolist(),
    'model_b_coef': model_b.params['feature12'],
    'model_b_p': model_b.pvalues['feature12'],
    'model_b_ci': model_b.conf_int().loc['feature12'].tolist(),
})
