import pandas as pd

df = pd.read_csv('amtl.csv')
print('fraction genus >1:', (df['genus']>1).mean())
print('fraction genus <0:', (df['genus']<0).mean())
print('min', df['genus'].min(), 'max', df['genus'].max())
