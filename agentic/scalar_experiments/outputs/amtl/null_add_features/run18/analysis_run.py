import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')
print('rows', len(_df))
print('columns', _df.columns.tolist())
print(_df[['num_amtl','sockets','age','prob_male','genus','tooth_class']].head())

# Check missing values in relevant cols
cols = ['num_amtl','sockets','age','prob_male','genus','tooth_class']
print('missing', _df[cols].isna().sum())

# Basic counts by genus
print(_df['genus'].value_counts())

# Fit binomial GLM: num_amtl / sockets
# Filter rows with valid sockets >0 and num_amtl>=0
_df2 = _df[cols].copy()
_df2 = _df2.dropna()
_df2 = _df2[_df2['sockets']>0]

# Create response as proportion with weights
_df2['amtl_prop'] = _df2['num_amtl'] / _df2['sockets']

model = smf.glm(
    formula='amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)',
    data=_df2,
    family=sm.families.Binomial(),
    freq_weights=_df2['sockets'],
)
res = model.fit()
print(res.summary())

# Extract genus effects vs Homo sapiens baseline? need set base.
# Relevel genus to Homo sapiens baseline
_df2['genus'] = pd.Categorical(_df2['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])
model2 = smf.glm(
    formula='amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)',
    data=_df2,
    family=sm.families.Binomial(),
    freq_weights=_df2['sockets'],
)
res2 = model2.fit()
print(res2.summary())

# Odds ratios for genus
params = res2.params
conf = res2.conf_int()
import numpy as np
or_params = np.exp(params)
or_conf = np.exp(conf)
print('ORs')
print(pd.DataFrame({'OR': or_params, 'OR_low': or_conf[0], 'OR_high': or_conf[1]}))
