import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.columns)
print(df.isna().sum())
print(df['feature8'].value_counts())
print(df['feature1'].value_counts())
# check ranges
print(df[['feature3','feature4']].describe())
# check any negatives or invalid
print('neg missing', (df['feature3']<0).sum())
print('missing > sockets', (df['feature3']>df['feature4']).sum())
# check unique sex values
print('sex unique', sorted(df['feature7'].unique()))
