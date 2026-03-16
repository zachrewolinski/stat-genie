import pandas as pd

df = pd.read_csv('amtl.csv')
# treat genus as missing count and age as sockets count
viol = (df['genus'] > df['age']).sum()
print('violations missing > sockets', viol)
print('min diff', (df['age'] - df['genus']).min())
print('any negative missing?', (df['genus']<0).sum())
