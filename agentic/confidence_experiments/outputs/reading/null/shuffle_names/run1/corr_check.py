import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# candidates for time columns
cols = ['adjusted_running_time','age','gender']
for col in cols:
    corr = df['running_time'].corr(df[col])
    print('corr running_time vs', col, corr)

# compute speed from num_words / adjusted_running_time (assuming ms)
calc_speed = df['num_words'] / (df['adjusted_running_time'] / 60000)
print('calc_speed stats', calc_speed.describe())
print('corr running_time vs calc_speed', df['running_time'].corr(calc_speed))

# using age as adjusted_running_time
calc_speed2 = df['num_words'] / (df['age'] / 60000)
print('corr running_time vs calc_speed2', df['running_time'].corr(calc_speed2))
