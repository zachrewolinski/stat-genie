import pandas as pd

df = pd.read_csv('amtl.csv')

sum_age = df.groupby('prob_male')['age'].sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()
pop = df.groupby('prob_male')['pop'].first()

print('Correlation sum_age vs num_amtl', sum_age.corr(num_amtl))
print('Correlation sum_age vs pop', sum_age.corr(pop))

print(pd.DataFrame({'sum_age':sum_age.head(), 'num_amtl':num_amtl.head(), 'pop':pop.head()}))
