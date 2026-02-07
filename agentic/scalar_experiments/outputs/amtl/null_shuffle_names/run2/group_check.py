import pandas as pd

df = pd.read_csv('amtl.csv')

for col in ['sockets','genus','age','pop','num_amtl','stdev_age','tooth_class','specimen']:
    vc = df.groupby('prob_male')[col].nunique().value_counts().sort_index()
    print(col, vc)
