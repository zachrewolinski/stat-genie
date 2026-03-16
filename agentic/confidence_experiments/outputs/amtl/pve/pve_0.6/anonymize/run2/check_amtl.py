import pandas as pd

df = pd.read_csv('amtl.csv')
print('min', df['feature3'].min())
print('max', df['feature3'].max())
print('negatives', (df['feature3'] < 0).sum())
print(df['feature3'].head())
