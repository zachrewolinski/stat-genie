import pandas as pd

df = pd.read_csv('amtl.csv')
# check if num_amtl constant per specimen
const = df.groupby('prob_male')['num_amtl'].nunique()
print(const.value_counts().head())
# check if pop constant per specimen
const_pop = df.groupby('prob_male')['pop'].nunique()
print('pop unique counts', const_pop.value_counts().head())
# check if genus numeric constant per specimen
const_gen = df.groupby('prob_male')['genus'].nunique()
print('genus unique counts', const_gen.value_counts().head())
