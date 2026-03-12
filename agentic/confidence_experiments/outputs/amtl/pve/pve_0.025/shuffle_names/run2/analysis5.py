import pandas as pd

df = pd.read_csv('amtl.csv')
print('num_amtl summary', df['num_amtl'].describe())
print('num_amtl unique sample', df['num_amtl'].nunique(), df['num_amtl'].head(10).tolist())
