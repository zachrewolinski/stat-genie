import pandas as pd

df = pd.read_csv('amtl.csv')
df['freq'] = df['num_amtl'] / df['sockets']
df['is_human'] = (df['genus'] == 'Homo sapiens')

mean_h = df[df['is_human']]['freq'].mean()
mean_n = df[~df['is_human']]['freq'].mean()
print('mean freq human', mean_h)
print('mean freq non-human', mean_n)
print('difference', mean_h-mean_n)

mean_n_amtl = df[~df['is_human']]['num_amtl'].mean()
mean_h_amtl = df[df['is_human']]['num_amtl'].mean()
print('mean num_amtl human', mean_h_amtl)
print('mean num_amtl non-human', mean_n_amtl)
print('difference num_amtl', mean_h_amtl-mean_n_amtl)

