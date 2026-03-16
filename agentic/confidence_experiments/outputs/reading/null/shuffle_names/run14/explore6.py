import pandas as pd

df = pd.read_csv('reading.csv')

total_from_running = df['running_time'] * df['num_words']

corr1 = total_from_running.corr(df['adjusted_running_time'])
corr2 = total_from_running.corr(df['age'])

print('corr total_from_running vs adjusted_running_time', corr1)
print('corr total_from_running vs age', corr2)

print('total_from_running stats', total_from_running.describe())
print('adjusted_running_time stats', df['adjusted_running_time'].describe())
print('age stats', df['age'].describe())
