import pandas as pd

df = pd.read_csv('amtl.csv')
# sum genus across classes per specimen
sum_genus = df.groupby('prob_male')['genus'].sum()
pop = df.groupby('prob_male')['pop'].first()
print('corr sum_genus vs pop', sum_genus.corr(pop))
print('sum_genus stats', sum_genus.describe())
