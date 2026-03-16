import pandas as pd
amtl = pd.read_csv('amtl.csv')
print('pop nunique within specimen:', amtl.groupby('prob_male')['pop'].nunique().describe())
print('fraction specimens with same pop across classes', (amtl.groupby('prob_male')['pop'].nunique()==1).mean())
print('age nunique within specimen:', amtl.groupby('prob_male')['age'].nunique().describe())
print('fraction specimens with same age across classes', (amtl.groupby('prob_male')['age'].nunique()==1).mean())
print('stdev_age nunique within specimen:', amtl.groupby('prob_male')['stdev_age'].nunique().describe())
print('fraction specimens with same stdev_age across classes', (amtl.groupby('prob_male')['stdev_age'].nunique()==1).mean())
