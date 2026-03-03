import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df['feature3'].describe())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())
rate = df['feature3'] / df['feature4']
print('rate describe', rate.describe())
print('rate min/max', rate.min(), rate.max())
print('num negative feature3', (df['feature3']<0).sum())
print('num feature3 > feature4', (df['feature3']>df['feature4']).sum())
print('num rate outside [0,1]', ((rate<0)|(rate>1)).sum())
print('genera', df['feature8'].value_counts())
print('tooth class', df['feature1'].value_counts())
