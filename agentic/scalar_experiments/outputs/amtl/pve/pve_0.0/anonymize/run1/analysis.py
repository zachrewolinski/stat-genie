import pandas as pd
import numpy as np

path = 'amtl.csv'

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# check unique values counts for feature3, feature4
print('feature3 min/max', df['feature3'].min(), df['feature3'].max())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())

# see if feature3 near integers when unstandardized? check closeness to integers
feat3 = df['feature3']
print('feature3 integer-like fraction', np.mean(np.isclose(feat3, np.round(feat3))))

# check if feature3 maybe proportion? compute correlation with feature4
print('corr feature3 vs feature4', df['feature3'].corr(df['feature4']))

# check groups
print(df['feature8'].value_counts())
print(df['feature1'].value_counts())

