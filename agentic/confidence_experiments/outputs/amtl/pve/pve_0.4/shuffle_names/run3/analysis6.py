import pandas as pd

df = pd.read_csv('amtl.csv')

# check variability within specimen id
for col in ['genus','age','pop','num_amtl','stdev_age']:
    grouped = df.groupby('prob_male')[col].nunique()
    print(col, 'median nunique within specimen', grouped.median(), 'max', grouped.max())

# check whether num_amtl constant across tooth_class within specimen
print('num_amtl unique counts', df.groupby('prob_male')['num_amtl'].nunique().value_counts().head())
print('age unique counts', df.groupby('prob_male')['age'].nunique().value_counts().head())
print('genus unique counts', df.groupby('prob_male')['genus'].nunique().value_counts().head())
