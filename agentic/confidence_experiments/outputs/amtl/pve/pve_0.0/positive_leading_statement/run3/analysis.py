import pandas as pd

# Load data
file_path = 'amtl.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.dtypes)
print('num_amtl unique sample:', df['num_amtl'].head(10).tolist())
print('num_amtl min max:', df['num_amtl'].min(), df['num_amtl'].max())
print('num_amtl integer check:', (df['num_amtl'] % 1 == 0).mean())
print('sockets min max:', df['sockets'].min(), df['sockets'].max())
print('sockets integer check:', (df['sockets'] % 1 == 0).mean())

print(df['genus'].value_counts())
print(df['tooth_class'].value_counts())
