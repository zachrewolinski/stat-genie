import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
neg = (df['feature3']<0).mean()
print('neg proportion', neg)
print('min feature3', df['feature3'].min())
print('max', df['feature3'].max())

# compute proportion missing using raw feature3/feature4
prop = df['feature3']/df['feature4']
print('prop min/max', prop.min(), prop.max())

