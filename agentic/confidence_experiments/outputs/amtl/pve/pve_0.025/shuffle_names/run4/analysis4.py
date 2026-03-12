import pandas as pd


df=pd.read_csv('amtl.csv')
# find a specimen with 3 rows
spec = df['prob_male'].unique()[10]
sub = df[df['prob_male']==spec]
print(spec)
print(sub)
print('num_amtl unique', sub['num_amtl'].unique())
print('pop unique', sub['pop'].unique())
print('age unique', sub['age'].unique())
print('genus unique', sub['genus'].unique())
print('stdev_age unique', sub['stdev_age'].unique())

# check for within-specimen variance in num_amtl or pop
print('num_amtl varies within specimen?', df.groupby('prob_male')['num_amtl'].nunique().value_counts().head())
print('pop varies within specimen?', df.groupby('prob_male')['pop'].nunique().value_counts().head())
print('age varies within specimen?', df.groupby('prob_male')['age'].nunique().value_counts().head())
print('genus varies within specimen?', df.groupby('prob_male')['genus'].nunique().value_counts().head())
