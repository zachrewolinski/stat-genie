import pandas as pd

df = pd.read_csv('amtl.csv')

# check within specimen (prob_male) variability
cols = ['genus','age','pop','num_amtl','stdev_age','tooth_class','specimen','sockets']

by_id = df.groupby('prob_male')

for col in ['genus','age','pop','num_amtl','stdev_age']:
    nunique = by_id[col].nunique()
    print(col, 'unique per id summary')
    print(nunique.describe())
    print('max unique', nunique.max())

# Check tooth_class per ID
print('tooth_class unique per id:')
print(by_id['sockets'].nunique().value_counts())

