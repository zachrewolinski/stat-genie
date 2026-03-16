import pandas as pd


df=pd.read_csv('amtl.csv')
# sum genus across classes per specimen
sum_genus = df.groupby('prob_male')['genus'].sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()
print('corr sum_genus with num_amtl', sum_genus.corr(num_amtl))
print('summary sum_genus', sum_genus.describe())
print('summary num_amtl', num_amtl.describe())

# check if num_amtl equals sum_genus maybe scaled
ratio = num_amtl / sum_genus
print('ratio num_amtl/sum_genus', ratio.describe())
