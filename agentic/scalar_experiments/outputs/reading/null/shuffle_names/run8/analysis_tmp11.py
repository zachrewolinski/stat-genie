import pandas as pd

df = pd.read_csv('reading.csv')
print('corr running_time vs adjusted_running_time', df['running_time'].corr(df['adjusted_running_time']))
print('corr running_time vs age', df['running_time'].corr(df['age']))
print('corr running_time vs gender', df['running_time'].corr(df['gender']))
