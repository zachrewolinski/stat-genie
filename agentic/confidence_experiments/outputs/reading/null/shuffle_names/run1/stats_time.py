import pandas as pd

df = pd.read_csv('reading.csv')
for col in ['adjusted_running_time','age','gender']:
    print('\n', col)
    print(df[col].describe())
    print('quantiles', df[col].quantile([0.01,0.05,0.5,0.95,0.99]))
