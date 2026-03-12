import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')
print('rows', len(_df))
print('num_amtl dtype', _df['num_amtl'].dtype)
print('num_amtl head', _df['num_amtl'].head().tolist())
print('num_amtl min/max', _df['num_amtl'].min(), _df['num_amtl'].max())

# Check if num_amtl appears integer-ish
num_amtl = _df['num_amtl']
print('num_amtl integer-like fraction', np.mean(np.isclose(num_amtl, np.round(num_amtl))))

# Check sockets
print('sockets unique', sorted(_df['sockets'].unique())[:10])

# Compute proportion missing
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']
print('amtl_rate summary', _df['amtl_rate'].describe())

# Species indicator: Homo sapiens vs non-human (Pan, Pongo, Papio)
_df['is_homo'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Basic OLS on amtl_rate
# controls: age, prob_male, tooth_class
model = smf.ols('amtl_rate ~ is_homo + age + prob_male + C(tooth_class)', data=_df).fit()
print(model.summary())

# Also OLS on num_amtl with same covariates and sockets as offset (or covariate)
model2 = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class) + sockets', data=_df).fit()
print(model2.summary())

# For binomial, attempt to use GLM if num_amtl appears count. If non-integer, skip.
if np.mean(np.isclose(num_amtl, np.round(num_amtl))) > 0.95:
    _df['num_amtl_round'] = np.round(num_amtl).astype(int)
    model3 = smf.glm('num_amtl_round / sockets ~ is_homo + age + prob_male + C(tooth_class)', data=_df,
                     family=sm.families.Binomial(), freq_weights=_df['sockets']).fit()
    print(model3.summary())
else:
    print('num_amtl not integer-like; skip binomial')
