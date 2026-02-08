import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
for col in ['genus','age']:
    print('\n', col)
    print(df.groupby('sockets')[col].agg(['min','max','mean']).round(3))

print('\nMissing > sockets? for candidate mappings')
miss = df['genus']
so = df['age']
print('genus>age count', int((miss>so).sum()))
miss = df['age']
so = df['genus']
print('age>genus count', int((miss>so).sum()))
