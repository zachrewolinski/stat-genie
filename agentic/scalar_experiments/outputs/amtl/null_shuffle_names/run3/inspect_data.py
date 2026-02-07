import pandas as pd


df = pd.read_csv('amtl.csv')
print('genus <= age proportion', (df['genus'] <= df['age']).mean())
print('max genus-age', (df['genus']-df['age']).max())
print('min genus-age', (df['genus']-df['age']).min())
print('num_amtl <= pop proportion', (df['num_amtl'] <= df['pop']).mean())
print('num_amtl <= age proportion', (df['num_amtl'] <= df['age']).mean())
print('pop <= 100', (df['pop']<=100).mean())
print('num_amtl <= 100', (df['num_amtl']<=100).mean())
print('correlations')
print(df[['genus','age','pop','num_amtl','stdev_age']].corr())
