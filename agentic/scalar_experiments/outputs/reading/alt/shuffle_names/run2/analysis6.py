import pandas as pd

df = pd.read_csv('reading.csv')
print('corr running_time vs adjusted', df['running_time'].corr(df['adjusted_running_time']))
print('corr running_time vs age', df['running_time'].corr(df['age']))
