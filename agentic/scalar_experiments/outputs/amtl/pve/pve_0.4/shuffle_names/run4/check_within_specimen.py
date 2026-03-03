import pandas as pd

df = pd.read_csv('amtl.csv')

# group by specimen id (prob_male column) and check number of unique values per variable
spec_col = 'prob_male'

for col in ['num_amtl','pop','age','genus','stdev_age','sockets','tooth_class','specimen']:
    unique_counts = df.groupby(spec_col)[col].nunique()
    print(col, 'unique per specimen: min', unique_counts.min(), 'max', unique_counts.max(), 'mean', unique_counts.mean())
    # proportion with single unique
    print('  prop single unique', (unique_counts==1).mean())
