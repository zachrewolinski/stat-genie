import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# compute derived wpm from adjusted_running_time (ms)

derived_wpm_adj = df['num_words'] / (df['adjusted_running_time']/60000.0)

derived_wpm_age = df['num_words'] / (df['age']/60000.0)

print('running_time summary', df['running_time'].describe())
print('derived_wpm_adj summary', derived_wpm_adj.describe())
print('derived_wpm_age summary', derived_wpm_age.describe())

print('corr running_time vs derived_wpm_adj', df['running_time'].corr(derived_wpm_adj))
print('corr running_time vs derived_wpm_age', df['running_time'].corr(derived_wpm_age))

# check if running_time equals derived_wpm_adj maybe? compute relative diff
rel_diff = (df['running_time'] - derived_wpm_adj)
print('diff adj stats', rel_diff.describe())
rel_diff_age = (df['running_time'] - derived_wpm_age)
print('diff age stats', rel_diff_age.describe())

# check if running_time equals derived_wpm_adj * some factor
ratio = df['running_time'] / derived_wpm_adj
print('ratio running_time/derived_wpm_adj', ratio.describe())
ratio_age = df['running_time'] / derived_wpm_age
print('ratio running_time/derived_wpm_age', ratio_age.describe())
