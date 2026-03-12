import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('amtl.csv')

df['genus_name'] = df['tooth_class']
df['tooth_class'] = df['sockets']
df['age_at_death'] = df['pop']
df['prob_male'] = df['stdev_age']
df['num_missing'] = df['num_amtl']
df['n_sockets'] = df['age']
df['amtl_rate'] = df['num_missing'] / df['n_sockets']

model = smf.ols('amtl_rate ~ C(genus_name) + age_at_death + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())
