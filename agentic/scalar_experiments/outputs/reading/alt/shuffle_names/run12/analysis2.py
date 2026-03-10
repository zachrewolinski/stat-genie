import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# Compute candidate reading speeds
# time in ms
age = df['age']
adj = df['adjusted_running_time']
scroll = df['gender']
words = df['retake_trial']

# wpm based on age (reading time minus scrolling)
wpm_age = words / (age / 60000)
# wpm based on adjusted
wpm_adj = words / (adj / 60000)
# wpm based on adj - scroll
wpm_adj_scroll = words / ((adj - scroll) / 60000)

for name, series in [('wpm_age', wpm_age), ('wpm_adj', wpm_adj), ('wpm_adj_scroll', wpm_adj_scroll)]:
    corr = series.corr(df['running_time'])
    print(name, 'corr with running_time', corr)

print('running_time describe', df['running_time'].describe())
print('wpm_age describe', wpm_age.describe())
print('wpm_adj describe', wpm_adj.describe())
print('wpm_adj_scroll describe', wpm_adj_scroll.describe())

# check which column equals words
# retake_trial repeated exactly 6 values. num_words 54 values. In info, num_words should be number of words.

# See correlation between running_time and 1/age? Maybe running_time is reading speed or time?
for col in ['adjusted_running_time','age','gender']:
    print('corr running_time with', col, df['running_time'].corr(df[col]))

# Check if running_time equals age/words? wpm or sec per word.
spw = age / words
print('spw corr', spw.corr(df['running_time']))

