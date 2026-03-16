import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('corr num_amtl vs sockets', df['num_amtl'].corr(df['sockets']))
# check if num_amtl proportion within 0-1 after maybe logistic? compute if num_amtl could be proportion by mapping? for now compute summary of num_amtl by sockets
print(df.groupby('sockets')['num_amtl'].mean().head())
