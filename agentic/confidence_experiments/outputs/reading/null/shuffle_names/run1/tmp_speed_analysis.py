import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate time columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('numeric columns', num_cols)

# show basic stats for selected columns
for col in num_cols:
    s = df[col]
    print('\n', col)
    print(s.describe())

# check if running_time is proportional to num_words / adjusted_running_time or age
if 'num_words' in df.columns:
    if 'adjusted_running_time' in df.columns:
        speed_words_per_ms = df['num_words'] / df['adjusted_running_time']
        print('\nnum_words/adjusted_running_time summary')
        print(speed_words_per_ms.describe())
        print('corr running_time vs num_words/adjusted_running_time', df['running_time'].corr(speed_words_per_ms))
    if 'age' in df.columns:
        speed_words_per_ms2 = df['num_words'] / df['age']
        print('\nnum_words/age summary')
        print(speed_words_per_ms2.describe())
        print('corr running_time vs num_words/age', df['running_time'].corr(speed_words_per_ms2))

# check correlation of running_time with adjusted_running_time
print('\nCorr running_time vs adjusted_running_time', df['running_time'].corr(df['adjusted_running_time']))
print('Corr running_time vs age', df['running_time'].corr(df['age']))
print('Corr running_time vs gender', df['running_time'].corr(df['gender']))

