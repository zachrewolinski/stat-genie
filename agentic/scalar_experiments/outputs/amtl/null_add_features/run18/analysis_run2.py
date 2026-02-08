import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

_df = pd.read_csv('amtl.csv')

# Basic validity
_df['invalid'] = _df['num_amtl'] > _df['sockets']
print('invalid rows', _df['invalid'].sum())

_df2 = _df[_df['num_amtl'] <= _df['sockets']].copy()
_df2 = _df2[_df2['sockets'] > 0].copy()

cols = ['num_amtl','sockets','age','prob_male','genus','tooth_class']
_df2 = _df2[cols].dropna()

# Set genus baseline as Homo sapiens
_df2['genus'] = pd.Categorical(_df2['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])

# Build endog as successes/failures
endog = np.column_stack((_df2['num_amtl'], _df2['sockets'] - _df2['num_amtl']))

# Design matrix using formula
formula = 'C(genus) + age + prob_male + C(tooth_class)'
exog = smf.glm(formula='num_amtl / sockets ~ ' + formula, data=_df2).exog

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()
print(res.summary())

# Map params to names
param_names = smf.glm(formula='num_amtl / sockets ~ ' + formula, data=_df2).exog_names
params = pd.Series(res.params, index=param_names)
conf = pd.DataFrame(res.conf_int(), index=param_names)

or_params = np.exp(params)
or_conf = np.exp(conf)
print('ORs')
print(pd.DataFrame({'OR': or_params, 'OR_low': or_conf[0], 'OR_high': or_conf[1]}))

# Also compute predicted amtl rates for each genus at mean covariates
mean_age = _df2['age'].mean()
mean_prob_male = _df2['prob_male'].mean()
common_tooth = _df2['tooth_class'].mode()[0]

# Build dataframe for prediction
pred_rows = []
for genus in ['Homo sapiens','Pan','Pongo','Papio']:
    pred_rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': common_tooth})

df_pred = pd.DataFrame(pred_rows)
# use same categorical levels
_df2['tooth_class'] = pd.Categorical(_df2['tooth_class'])
df_pred['tooth_class'] = pd.Categorical(df_pred['tooth_class'], categories=_df2['tooth_class'].cat.categories)

design = smf.glm(formula='num_amtl / sockets ~ ' + formula, data=_df2).model.data.design_info
exog_pred = design.transform(df_pred)
linpred = exog_pred @ res.params
pred_prob = 1 / (1 + np.exp(-linpred))
print('predicted rates at mean covariates')
for g, p in zip(df_pred['genus'], pred_prob):
    print(g, p)
