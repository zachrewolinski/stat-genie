import pandas as pd
import statsmodels.formula.api as smf

amtl = pd.read_csv('amtl.csv')
amtl['is_human'] = (amtl['genus'] == 'Homo sapiens').astype(int)
amtl['tooth_class'] = amtl['tooth_class'].astype('category')

model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class) + sockets', data=amtl).fit(cov_type='cluster', cov_kwds={'groups': amtl['specimen']})
print(model.summary())
