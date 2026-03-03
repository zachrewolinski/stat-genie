import pandas as pd

df = pd.read_csv('amtl.csv')

sum_genus = df.groupby('prob_male')['genus'].sum()
print(sum_genus.describe())

