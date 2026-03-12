import pandas as pd
amtl = pd.read_csv('amtl.csv')
print(amtl.groupby('sockets')['age'].describe())
print('\nmean age by sockets', amtl.groupby('sockets')['age'].mean())
print('\nmean num_amtl by sockets', amtl.groupby('sockets')['num_amtl'].mean())
print('\nmean genus by sockets', amtl.groupby('sockets')['genus'].mean())
