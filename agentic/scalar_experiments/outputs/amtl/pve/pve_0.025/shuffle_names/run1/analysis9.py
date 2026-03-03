import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

df['is_human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

df['rate'] = df['genus'] / df['age']

print(df.groupby('is_human')[['genus','rate']].mean())
print(df.groupby('is_human')[['genus','rate']].median())

