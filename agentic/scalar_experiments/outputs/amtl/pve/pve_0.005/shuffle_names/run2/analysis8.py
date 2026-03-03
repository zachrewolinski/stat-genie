import pandas as pd

df = pd.read_csv('amtl.csv')

sum_age = df.groupby('prob_male')['age'].sum()
print(sum_age.describe())

