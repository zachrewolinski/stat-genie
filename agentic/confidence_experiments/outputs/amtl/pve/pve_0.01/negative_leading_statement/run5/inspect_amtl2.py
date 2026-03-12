import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print('corr num_amtl vs sockets', df['num_amtl'].corr(df['sockets']))
print('mean sockets', df['sockets'].mean())
print(df[['num_amtl','sockets']].head())

# Check if num_amtl could be standardized within sockets? compute expected counts by rounding inverse z? no

