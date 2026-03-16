import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')
df['log_deaths'] = np.log1p(df['feature8'])

model = smf.ols('log_deaths ~ feature12 + feature7 + feature5 + feature13', data=df).fit(cov_type='HC3')
print(model.summary())
