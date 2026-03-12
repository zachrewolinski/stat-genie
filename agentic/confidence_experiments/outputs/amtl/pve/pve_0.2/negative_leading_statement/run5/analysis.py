import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')
print('num_amtl mean', df['num_amtl'].mean(), 'std', df['num_amtl'].std())
print('num_amtl min', df['num_amtl'].min(), 'max', df['num_amtl'].max())
print('num_amtl unique approx integer?', np.allclose(df['num_amtl'], df['num_amtl'].round()))
print('sockets min/max', df['sockets'].min(), df['sockets'].max())
print('num_amtl head', df['num_amtl'].head().tolist())

u = np.sort(df['num_amtl'].unique())
print('unique count', len(u))
print('diff median', np.median(np.diff(u)))

print(df.groupby('genus')['num_amtl'].describe())
