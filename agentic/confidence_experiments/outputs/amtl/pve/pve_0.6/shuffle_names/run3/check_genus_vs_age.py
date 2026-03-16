import pandas as pd


df = pd.read_csv('amtl.csv')
print('share genus<=age', (df['genus'] <= df['age']).mean())
print('share genus>=0', (df['genus'] >= 0).mean())
print('min genus', df['genus'].min())

