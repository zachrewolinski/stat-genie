import pandas as pd

df = pd.read_csv('amtl.csv')

# Check if num_amtl <= age
print('num_amtl <= age proportion', (df['num_amtl'] <= df['age']).mean())
print('num_amtl <= pop proportion', (df['num_amtl'] <= df['pop']).mean())
print('genus <= age proportion', (df['genus'] <= df['age']).mean())
