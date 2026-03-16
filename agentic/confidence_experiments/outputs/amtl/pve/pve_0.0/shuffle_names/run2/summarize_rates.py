import pandas as pd

df = pd.read_csv('amtl.csv')

df['genus_name'] = df['tooth_class']
df['tooth_class'] = df['sockets']
df['age_at_death'] = df['pop']
df['prob_male'] = df['stdev_age']
df['num_missing'] = df['num_amtl']
df['n_sockets'] = df['age']
df['amtl_rate'] = df['num_missing'] / df['n_sockets']

print(df.groupby('genus_name')['amtl_rate'].describe())
