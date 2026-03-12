import pandas as pd

df = pd.read_csv('reading.csv')

calc_speed1 = df['num_words'] / df['adjusted_running_time'] * 60000
calc_speed2 = df['num_words'] / df['running_time'] * 60000

print('corr calc_speed1 vs running_time', calc_speed1.corr(df['running_time']))
print('corr calc_speed2 vs adjusted_running_time', calc_speed2.corr(df['adjusted_running_time']))

print('calc_speed1 stats', calc_speed1.describe())
print('calc_speed2 stats', calc_speed2.describe())
