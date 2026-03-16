import pandas as pd

df = pd.read_csv('amtl.csv')
per = df.groupby('prob_male')
sum_age = per['age'].sum()
pop = per['pop'].first()
print('corr pop vs sum_age', pop.corr(sum_age))
print('mean diff', (pop - sum_age).mean())
print('std diff', (pop - sum_age).std())
print((pop - sum_age).describe())
