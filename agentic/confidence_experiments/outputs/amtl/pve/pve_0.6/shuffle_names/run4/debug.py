import pandas as pd

df = pd.read_csv('amtl.csv')
df = df.rename(columns={'genus': 'amtl','pop': 'age_at_death','stdev_age':'prob_male','sockets':'tooth_class','tooth_class':'genus'})

df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens','Pan','Papio','Pongo'])

df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior','Posterior','Premolar'])

print(type(df['genus']))
print(df['genus'].shape)
print(df['genus'].head())
print(df['genus'].dtype)
print(df['genus'].cat.categories)

print(type(df['tooth_class']))
print(df['tooth_class'].shape)
print(df['tooth_class'].head())
print(df['tooth_class'].dtype)

