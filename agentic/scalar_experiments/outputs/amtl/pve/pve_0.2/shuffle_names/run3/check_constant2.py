import pandas as pd

df = pd.read_csv('amtl.csv')
for col in ['stdev_age','age','pop','num_amtl','genus']:
    counts = df.groupby('prob_male')[col].nunique()
    print(col, counts.value_counts().head())
