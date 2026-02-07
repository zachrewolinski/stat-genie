import pandas as pd

df = pd.read_csv('amtl.csv')
df = df.rename(columns={'genus':'num_missing','age':'num_sockets','pop':'age_est','num_amtl':'age_stdev','stdev_age':'prob_male','tooth_class':'genus_group','sockets':'tooth_class'})
valid = df['num_missing'] <= df['num_sockets']
clean = df.loc[valid].copy()
print(clean.dtypes)
print(clean['tooth_class'].head())
print(type(clean['tooth_class']))
print(clean['tooth_class'].map(type).unique()[:5])
