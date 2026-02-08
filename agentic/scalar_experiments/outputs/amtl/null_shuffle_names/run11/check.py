import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
missing = df['genus']
sockets = df['age']
print('sockets min', sockets.min())
print('sockets <=0', (sockets<=0).sum())
print('missing <0', (missing<0).sum())
print('missing > sockets', (missing> sockets).sum())
print('missing==0', (missing==0).sum())

# rows where sockets==0
print(df[sockets<=0].head())

# any NaN
print('NaN counts', df[['genus','age','pop','stdev_age']].isna().sum())
