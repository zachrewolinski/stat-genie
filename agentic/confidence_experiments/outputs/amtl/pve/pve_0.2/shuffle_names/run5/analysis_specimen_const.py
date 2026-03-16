import pandas as pd

df = pd.read_csv('amtl.csv')
for col in ['tooth_class','specimen','sockets']:
    print(col, df.groupby('prob_male')[col].nunique().value_counts().sort_index().head())
