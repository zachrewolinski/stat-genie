import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

_df = pd.read_csv('amtl.csv')
_df = _df[_df['num_amtl'] <= _df['sockets']].copy()
_df = _df[_df['sockets'] > 0].copy()

cols = ['num_amtl','sockets','age','prob_male','genus','tooth_class']
_df = _df[cols].dropna()

_df['genus'] = pd.Categorical(_df['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])
_df['tooth_class'] = pd.Categorical(_df['tooth_class'])

_df['amtl_prop'] = _df['num_amtl'] / _df['sockets']

formula = 'amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets'],
)
res = model.fit()
print(res.summary())

params = res.params
conf = res.conf_int()
or_params = np.exp(params)
or_conf = np.exp(conf)
print('ORs')
print(pd.DataFrame({'OR': or_params, 'OR_low': or_conf[0], 'OR_high': or_conf[1]}))

# Predictions at mean covariates and most common tooth_class
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()
common_tooth = _df['tooth_class'].mode()[0]

pred_rows = []
for genus in ['Homo sapiens','Pan','Pongo','Papio']:
    pred_rows.append({'genus': genus, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': common_tooth})

df_pred = pd.DataFrame(pred_rows)
# ensure categories match
_df_genus_cat = _df['genus'].cat.categories
_df_tooth_cat = _df['tooth_class'].cat.categories

df_pred['genus'] = pd.Categorical(df_pred['genus'], categories=_df_genus_cat)
df_pred['tooth_class'] = pd.Categorical(df_pred['tooth_class'], categories=_df_tooth_cat)

pred_prob = res.predict(df_pred)
print('predicted rates at mean covariates')
for g, p in zip(df_pred['genus'], pred_prob):
    print(g, p)
