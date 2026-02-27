import pandas as pd

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('unique feature3', df['feature3'].unique()[:20], 'nunique', df['feature3'].nunique())
print('unique feature4 min max', df['feature4'].min(), df['feature4'].max())
print('unique feature4', sorted(df['feature4'].unique())[:20])
print('feature1 unique', df['feature1'].unique())
print('feature8 unique', df['feature8'].unique())

# check if feature3 * feature4 is integer maybe missing count
import numpy as np
mult = df['feature3'] * df['feature4']
# check close to integer
ints = np.isclose(mult, np.round(mult))
print('mult integer fraction', ints.mean())
print('mult unique rounded sample', np.unique(np.round(mult))[:20])

# if feature3 is proportion, check distribution
print(df['feature3'].value_counts().head(10))

# check if feature3 is maybe count/observable for each class? compute missing count (round)
df['missing_count'] = np.round(mult).astype(int)
print(df[['feature3','feature4','missing_count']].head(10))
print('missing_count unique', sorted(df['missing_count'].unique())[:20])

