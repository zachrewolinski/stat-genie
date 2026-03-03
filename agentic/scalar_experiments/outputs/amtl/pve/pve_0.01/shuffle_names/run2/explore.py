import pandas as pd

df = pd.read_csv('amtl.csv')

print('sockets unique:', df['sockets'].unique())
print('tooth_class unique:', df['tooth_class'].unique())
print('specimen unique sample:', df['specimen'].unique()[:10])
print('prob_male unique sample:', df['prob_male'].unique()[:10])

# check ranges
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print(col, df[col].min(), df[col].max())

# check if genus maybe missing teeth? examine counts
print(df[['genus','age','pop','num_amtl','stdev_age']].describe())
