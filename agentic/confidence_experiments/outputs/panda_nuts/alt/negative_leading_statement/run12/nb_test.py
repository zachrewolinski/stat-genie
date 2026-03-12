import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('panda_nuts.csv')

offset = np.log(df['seconds'])

try:
    nb_model = smf.negativebinomial('nuts_opened ~ age + C(sex) + C(help)', data=df, offset=offset)
    nb_res = nb_model.fit(disp=0)
    print(nb_res.summary())
    print('alpha', nb_res.params.get('alpha', None))
except Exception as e:
    print('negativebinomial formula failed:', e)

# Try discrete model directly
try:
    import statsmodels.discrete.discrete_model as smd
    y = df['nuts_opened']
    X = sm.add_constant(pd.get_dummies(df[['age','sex','help']], drop_first=True))
    nb2 = smd.NegativeBinomial(y, X, offset=offset)
    nb2_res = nb2.fit(disp=0)
    print(nb2_res.summary())
    print('alpha', nb2_res.params[-1])
except Exception as e:
    print('discrete NegativeBinomial failed:', e)
