import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','age','pop','num_amtl','stdev_age']:
    print('\n', col)
    print(df.groupby('tooth_class')[col].agg(['mean','std','min','max']))

# also by sockets
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print('\n', col, 'by sockets')
    print(df.groupby('sockets')[col].agg(['mean','std','min','max']))

# check correlation between potential counts
print('\nCorrelations:')
print(df[['genus','age','pop','num_amtl','stdev_age']].corr())

