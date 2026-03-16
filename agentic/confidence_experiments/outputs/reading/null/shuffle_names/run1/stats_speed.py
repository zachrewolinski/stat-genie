import pandas as pd

df = pd.read_csv('reading.csv')
print(df['running_time'].describe())
print('quantiles', df['running_time'].quantile([0.01,0.05,0.5,0.95,0.99]))
