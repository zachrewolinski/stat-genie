import pandas as pd

_df = pd.read_csv('amtl.csv')

counts = _df['prob_male'].value_counts()
print('Rows per specimen (min,max,mean):', counts.min(), counts.max(), counts.mean())

for col in ['age','pop','num_amtl','stdev_age','genus']:
    nunq = _df.groupby('prob_male')[col].nunique()
    print(col, 'unique per specimen: min', nunq.min(),'max',nunq.max())

sample_ids = counts.head(3).index.tolist()
for sid in sample_ids:
    sub = _df[_df['prob_male']==sid][['sockets','age','pop','num_amtl','stdev_age','genus']]
    print('\n', sid)
    print(sub)

