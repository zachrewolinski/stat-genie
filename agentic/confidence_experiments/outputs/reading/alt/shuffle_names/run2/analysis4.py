import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# compute wpm if adjusted_running_time is centiseconds
wpm_centi = df['num_words'] * 6000 / df['adjusted_running_time']

print('wpm_centi summary', wpm_centi.describe())
print('corr running_time vs wpm_centi', df['running_time'].corr(wpm_centi))
print('ratio running_time / wpm_centi', (df['running_time']/wpm_centi).describe())

# compute wpm if adjusted_running_time is milliseconds
wpm_ms = df['num_words'] * 60000 / df['adjusted_running_time']
print('wpm_ms summary', wpm_ms.describe())
print('corr running_time vs wpm_ms', df['running_time'].corr(wpm_ms))

# compute wpm if adjusted_running_time is seconds
wpm_sec = df['num_words'] * 60 / df['adjusted_running_time']
print('wpm_sec summary', wpm_sec.describe())
print('corr running_time vs wpm_sec', df['running_time'].corr(wpm_sec))
