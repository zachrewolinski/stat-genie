import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
import patsy

# load data

df = pd.read_csv('panda_nuts.csv')
renamed = df.rename(columns={
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'seconds': 'helped',
    'help': 'nuts_opened',
    'chimpanzee': 'duration_sec',
    'age': 'chimp_id',
    'sex': 'hammer_type',
})
renamed['sex'] = renamed['sex'].astype('category')
renamed['helped'] = renamed['helped'].astype('category')
renamed['hammer_type'] = renamed['hammer_type'].astype('category')
renamed = renamed[renamed['duration_sec'] > 0].copy()

# design matrices
formula = 'nuts_opened ~ age_years + C(sex) + C(helped)'

y, X = patsy.dmatrices(formula, data=renamed, return_type='dataframe')

# offset is log duration
offset = np.log(renamed['duration_sec'])

nb_model = NegativeBinomial(y, X, offset=offset)
nb_res = nb_model.fit(disp=False)
print(nb_res.summary())

# rate ratios
params = nb_res.params
conf = nb_res.conf_int()
rr = np.exp(params)
ci_low = np.exp(conf[0])
ci_high = np.exp(conf[1])
print('Rate ratios')
print(pd.DataFrame({'RR': rr, 'CI_low': ci_low, 'CI_high': ci_high}))

