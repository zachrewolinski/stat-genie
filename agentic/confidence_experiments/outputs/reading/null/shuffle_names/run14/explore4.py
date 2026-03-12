import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# compute speed from num_words and adjusted_running_time (age)
calc_speed = df['num_words'] / df['age'] * 60000

# compare to running_time column
corr = calc_speed.corr(df['running_time'])
print('corr calc_speed vs running_time', corr)

# simple summary
print(calc_speed.describe())
print(df['running_time'].describe())
