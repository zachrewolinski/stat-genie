import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
neg = (df['genus'] < 0).sum()
print('negative genus', neg)
print('min genus', df['genus'].min())
print('min genus by tooth_class', df.groupby('tooth_class')['genus'].min())

# proportion
prop = df['genus'] / df['age']
print('prop min', prop.min(), 'max', prop.max())
print('prop negative', (prop < 0).sum())

