import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

print('corr adjusted_running_time vs age', df['adjusted_running_time'].corr(df['age']))
print('corr adjusted_running_time vs gender', df['adjusted_running_time'].corr(df['gender']))
print('corr age vs gender', df['age'].corr(df['gender']))

