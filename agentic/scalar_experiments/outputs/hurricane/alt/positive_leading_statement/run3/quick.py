import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['alldeaths'])

m_simple = smf.ols('log_deaths ~ masfem', data=_df).fit(cov_type='HC3')
mg_simple = smf.ols('log_deaths ~ gender_mf', data=_df).fit(cov_type='HC3')
print(m_simple.summary().tables[1])
print(mg_simple.summary().tables[1])
