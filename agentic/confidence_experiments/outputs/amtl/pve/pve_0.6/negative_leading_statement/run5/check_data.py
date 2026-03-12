import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# check if num_amtl is integer-ish and within [0, sockets]
within = ((df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])).mean()
print('within_0_sockets_fraction', within)
# check near integer
near_int = (np.abs(df['num_amtl'] - df['num_amtl'].round()) < 1e-6).mean()
print('near_int_fraction', near_int)
print('num_amtl min max', df['num_amtl'].min(), df['num_amtl'].max())
# check unique values sample
print('num_amtl unique sample', df['num_amtl'].sort_values().head(10).tolist())
print('num_amtl unique sample high', df['num_amtl'].sort_values().tail(10).tolist())
# check ratio correlation with sockets
ratio = df['num_amtl'] / df['sockets']
print('ratio min max', ratio.min(), ratio.max())
print('ratio percentiles', ratio.quantile([0.01,0.05,0.5,0.95,0.99]).to_dict())
# check if num_amtl appears z-scored: mean, std
print('num_amtl mean', df['num_amtl'].mean(), 'std', df['num_amtl'].std())
