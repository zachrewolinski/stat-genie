import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate columns for time on page and adjusted time
for base_col in ['adjusted_running_time','age']:
    for scroll_col in ['gender']:
        other = 'age' if base_col == 'adjusted_running_time' else 'adjusted_running_time'
        diff = df[base_col] - df[scroll_col]
        # compare with other
        corr = diff.corr(df[other])
        print(f"{base_col} - {scroll_col} corr with {other}: {corr}")
        print('diff summary', diff.describe())

# check if adjusted_running_time approx equals age + gender
sum_corr = (df['age'] + df['gender']).corr(df['adjusted_running_time'])
print('age+gender corr adjusted_running_time', sum_corr)
print('age+gender diff summary', (df['adjusted_running_time'] - (df['age'] + df['gender'])).describe())
