import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('panda_nuts.csv')

df['age_years'] = df['age']

df['sex_mf'] = df['nuts_opened']

df['helped'] = df['seconds'].str.upper().map({'Y': 1, 'N': 0})

df['nuts_opened_count'] = df['help']

df['duration_sec'] = df['chimpanzee']

df['log_duration'] = np.log(df['duration_sec'])

model = smf.glm(
    formula='nuts_opened_count ~ age_years + C(sex_mf) + helped',
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=df['log_duration']
).fit()

params = model.params
conf = model.conf_int()

print('NB coefficients and rate ratios')
for param in params.index:
    if param == 'Intercept':
        continue
    rr = np.exp(params[param])
    low = np.exp(conf.loc[param,0])
    high = np.exp(conf.loc[param,1])
    print(param, 'coef', params[param], 'p', model.pvalues[param], 'RR', rr, 'CI', low, high)

