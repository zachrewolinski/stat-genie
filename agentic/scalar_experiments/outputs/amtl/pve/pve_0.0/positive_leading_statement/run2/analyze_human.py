import pandas as pd
import statsmodels.formula.api as smf

# load data

df = pd.read_csv('amtl.csv')

# create is_human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

df['tooth_class'] = df['tooth_class'].astype('category')

model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())

coef = model.params['is_human']
se = model.bse['is_human']
p = model.pvalues['is_human']
ci = model.conf_int().loc['is_human'].tolist()
print('is_human coef', coef, 'se', se, 'p', p, 'ci', ci)
