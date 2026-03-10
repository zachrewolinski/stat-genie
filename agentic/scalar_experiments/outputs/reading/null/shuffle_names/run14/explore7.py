import pandas as pd

df = pd.read_csv('reading.csv')

num_words = df['retake_trial']  # likely num_words
adj_time = df['age']            # likely adjusted_running_time
run_time = df['adjusted_running_time'] # likely running_time

calc_speed_adj = num_words / adj_time * 60000
calc_speed_run = num_words / run_time * 60000

print('corr speed_adj vs running_time_col', calc_speed_adj.corr(df['running_time']))
print('corr speed_run vs running_time_col', calc_speed_run.corr(df['running_time']))

print('calc_speed_adj stats', calc_speed_adj.describe())
print('calc_speed_run stats', calc_speed_run.describe())
print('running_time_col stats', df['running_time'].describe())
