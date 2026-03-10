import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
print(corr['running_time'].sort_values())

# correlation between running_time and num_words
print('corr running_time vs num_words', df['running_time'].corr(df['num_words']))

# compute wpm assuming running_time is seconds
wpm_rt = df['num_words'] / df['running_time'] * 60
print('wpm_rt stats', wpm_rt.describe())

# check correlation between wpm_rt and other time columns
for col in ['adjusted_running_time','age']:
    print('corr wpm_rt vs', col, wpm_rt.corr(df[col]))

