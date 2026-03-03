import pandas as pd
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)
print('columns', df.columns.tolist())
print(df.head())
print(df[['num_amtl','sockets','age','prob_male','genus','tooth_class']].describe(include='all'))

# Check num_amtl range and if integer
print('num_amtl unique sample', df['num_amtl'].head(10).tolist())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets min/max', df['sockets'].min(), df['sockets'].max())

# Check if num_amtl and sockets are integers
print('num_amtl integer-like', np.all(np.isclose(df['num_amtl'], df['num_amtl'].round())))
print('sockets integer-like', np.all(np.isclose(df['sockets'], df['sockets'].round())))

# Check rows where num_amtl negative
neg = df[df['num_amtl'] < 0]
print('negative num_amtl count', len(neg))
print(neg[['num_amtl','sockets','genus','tooth_class']].head())

