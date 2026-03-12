import pandas as pd
import numpy as np

path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# Check if feature3 looks integer-like
f3 = df['feature3']
print('feature3 integer-like proportion', np.mean(np.isclose(f3, np.round(f3))))
print('feature3 min/max', f3.min(), f3.max())

# check per genus summary of feature3 and feature4
print(df.groupby('feature8')[['feature3','feature4','feature5','feature7']].agg(['mean','std','min','max']).round(3))

# check possible relation: feature3 vs feature4
print('corr feature3 vs feature4', df[['feature3','feature4']].corr().iloc[0,1])

# Check if any negative feature3
print('neg feature3 count', (f3<0).sum())

# maybe feature3 standardized: mean, std
print('feature3 mean,std', f3.mean(), f3.std())

