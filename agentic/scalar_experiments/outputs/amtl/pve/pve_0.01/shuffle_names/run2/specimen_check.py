import pandas as pd

df = pd.read_csv('amtl.csv')

# check how many rows per specimen
spec_counts = df['prob_male'].value_counts()
print('rows per specimen unique', spec_counts.unique()[:10])

# check within specimen variation for numeric cols
cols = ['genus','age','pop','num_amtl','stdev_age']
for col in cols:
    var_within = df.groupby('prob_male')[col].nunique().value_counts().sort_index()
    print(col, var_within.head())
