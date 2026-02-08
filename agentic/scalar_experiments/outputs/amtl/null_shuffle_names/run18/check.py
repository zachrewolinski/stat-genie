import pandas as pd

df = pd.read_csv('amtl.csv')
# check if genus <= age
print('genus<=age proportion', (df['genus'] <= df['age']).mean())
print('max genus-age', (df['genus'] - df['age']).max())
print('min genus-age', (df['genus'] - df['age']).min())
# if num_amtl maybe sockets? check num_amtl vs age or genus
print('num_amtl<=pop proportion', (df['num_amtl'] <= df['pop']).mean())
print('num_amtl<=age proportion', (df['num_amtl'] <= df['age']).mean())
print('num_amtl<=genus proportion', (df['num_amtl'] <= df['genus']).mean())
# check if age <=? maybe sockets count should be small integer; age column is int 2-14.
print('age unique', sorted(df['age'].unique()))
