import pandas as pd

_df = pd.read_csv('reading.csv')

# check if adjusted_running_time ~= age + gender
s = _df['age'] + _df['gender']

print('mean diff', (s - _df['adjusted_running_time']).abs().mean())
print('median diff', (s - _df['adjusted_running_time']).abs().median())
print('max diff', (s - _df['adjusted_running_time']).abs().max())
print('corr age+gender vs adjusted', s.corr(_df['adjusted_running_time']))

# maybe adjusted = age - gender
s2 = _df['age'] - _df['gender']
print('mean diff age-gender', (s2 - _df['adjusted_running_time']).abs().mean())
print('corr age-gender vs adjusted', s2.corr(_df['adjusted_running_time']))
