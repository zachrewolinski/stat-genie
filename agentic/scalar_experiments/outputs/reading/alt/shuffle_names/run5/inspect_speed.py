import pandas as pd
import numpy as np

_df = pd.read_csv('reading.csv')

# Potential time columns
print('time-like columns summary:')
for col in ['adjusted_running_time','age','gender','running_time']:
    s = _df[col]
    print(col, s.min(), s.median(), s.mean(), s.max())

# compute candidate speed from num_words and adjusted_running_time (ms)
# speed_wpm = words / (time_minutes)
_speed_adj = _df['num_words'] / (_df['adjusted_running_time'] / 60000)
_speed_age = _df['num_words'] / (_df['age'] / 60000)
_speed_gender = _df['num_words'] / (_df['gender'] / 60000)

# correlation with running_time column
for name, sp in [('speed_adj', _speed_adj), ('speed_age', _speed_age), ('speed_gender', _speed_gender)]:
    corr = _df['running_time'].corr(sp)
    print('corr running_time vs', name, corr)

# Check if any column matches computed speed (approx)
for name, sp in [('speed_adj', _speed_adj), ('speed_age', _speed_age), ('speed_gender', _speed_gender)]:
    print(name, sp.describe(percentiles=[0.01,0.05,0.5,0.95,0.99]))

# maybe running_time is time (ms) rather than speed; check correlation with adjusted_running_time
print('corr running_time vs adjusted_running_time', _df['running_time'].corr(_df['adjusted_running_time']))
print('corr running_time vs age', _df['running_time'].corr(_df['age']))

