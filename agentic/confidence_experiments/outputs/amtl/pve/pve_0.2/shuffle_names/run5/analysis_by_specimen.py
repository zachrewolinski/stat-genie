import pandas as pd

df = pd.read_csv('amtl.csv')
# check if num_amtl and pop are constant within specimen id
for col in ['num_amtl','pop','stdev_age','genus','age']:
    nunique = df.groupby('prob_male')[col].nunique().describe()
    print(col, nunique)

# check within specimen variation for num_amtl
print('num_amtl per specimen unique counts (value counts):')
print(df.groupby('prob_male')['num_amtl'].nunique().value_counts().sort_index())
print('pop per specimen unique counts:')
print(df.groupby('prob_male')['pop'].nunique().value_counts().sort_index())
print('stdev_age per specimen unique counts:')
print(df.groupby('prob_male')['stdev_age'].nunique().value_counts().sort_index())
