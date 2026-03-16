import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)

# summary of feature4/5/6
print(df[['feature4','feature5','feature6']].describe())

# correlations
for col in ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature19']:
    corr = df['feature20'].corr(df[col])
    print('corr feature20 vs', col, corr)

# check if feature20 maybe rate? compute words per second or per minute etc
words = df['feature7']

speed_wpm_total = words / (df['feature4'] / 60000.0)
speed_wpm_noscroll = words / (df['feature5'] / 60000.0)

speed_wps_total = words / (df['feature4'] / 1000.0)
speed_wps_noscroll = words / (df['feature5'] / 1000.0)

# compare correlation with feature20
for name, series in [
    ('wpm_total', speed_wpm_total),
    ('wpm_noscroll', speed_wpm_noscroll),
    ('wps_total', speed_wps_total),
    ('wps_noscroll', speed_wps_noscroll),
]:
    print('corr feature20 vs', name, df['feature20'].corr(series))

# try to see if feature20 roughly equal to 60* words / time? that is wpm
# compute ratio feature20 / speed
for name, series in [('wpm_total', speed_wpm_total), ('wpm_noscroll', speed_wpm_noscroll)]:
    ratio = df['feature20'] / series
    print('ratio feature20 /', name, ratio.describe())

# check if feature20 maybe time per word? (ms per word)
time_per_word = df['feature5'] / words
print('corr feature20 vs time_per_word', df['feature20'].corr(time_per_word))
print('feature20 * time_per_word', (df['feature20']*time_per_word).describe())

