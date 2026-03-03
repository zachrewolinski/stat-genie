import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# human indicator

df['human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

# OLS for num_amtl
formula = 'num_amtl ~ human + pop + stdev_age + C(sockets)'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prob_male']})
print(model.summary())

# deduplicate by specimen (prob_male)
df_spec = df.drop_duplicates(subset=['prob_male'])
model2 = smf.ols('num_amtl ~ human + pop + stdev_age', data=df_spec).fit(cov_type='HC3')
print('\nDedup model:')
print(model2.summary())

# model with genus categories
df_spec['tooth_class'] = df_spec['tooth_class']
model3 = smf.ols('num_amtl ~ C(tooth_class) + pop + stdev_age', data=df_spec).fit(cov_type='HC3')
print('\nDedup with genus categories:')
print(model3.summary())

