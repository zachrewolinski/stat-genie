import pandas as pd

df = pd.read_csv('amtl.csv')
print('genus <= age:', (df['genus'] <= df['age']).mean(), (df['genus'] > df['age']).sum())
print('num_amtl <= age:', (df['num_amtl'] <= df['age']).mean(), (df['num_amtl'] > df['age']).sum())
print('genus range', df['genus'].min(), df['genus'].max())
print('age range', df['age'].min(), df['age'].max())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
