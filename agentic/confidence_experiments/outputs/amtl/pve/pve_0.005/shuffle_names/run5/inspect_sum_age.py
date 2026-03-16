import pandas as pd

df = pd.read_csv('amtl.csv')
# sum age across classes per specimen
sum_age = df.groupby('prob_male')['age'].sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()
print('sum_age stats', sum_age.describe())
print('corr sum_age vs num_amtl', sum_age.corr(num_amtl))
