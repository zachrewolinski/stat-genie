import pandas as pd
amtl = pd.read_csv('amtl.csv')
print('genus negative fraction', (amtl['genus']<0).mean())
print('min negative rows', amtl[amtl['genus']<0].head())
