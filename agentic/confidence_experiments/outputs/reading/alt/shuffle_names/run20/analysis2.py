import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 50)

df = pd.read_csv('reading.csv')

print('unique counts for categorical/object:')
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, df[col].nunique(), df[col].unique()[:10])

print('\nvalue counts for device (numeric):')
print(df['device'].value_counts(dropna=False).sort_index())
print('\nvalue counts for dyslexia:')
print(df['dyslexia'].value_counts(dropna=False).sort_index())
print('\nvalue counts for dyslexia_bin:')
print(df['dyslexia_bin'].value_counts(dropna=False).sort_index())
print('\nvalue counts for correct_rate:')
print(df['correct_rate'].value_counts(dropna=False).sort_index())

print('\nsummary of running_time:')
print(df['running_time'].describe())
print('\nsummary of adjusted_running_time:')
print(df['adjusted_running_time'].describe())
print('\nsummary of age:')
print(df['age'].describe())

# compute derived reading speed from num_words / adjusted_running_time (ms) * 60000
speed_calc = df['num_words'] / df['adjusted_running_time'] * 60000
print('\ncalculated wpm describe:')
print(speed_calc.describe())

# correlation between running_time and calc speed
print('\ncorrelation running_time with calc speed:', np.corrcoef(df['running_time'], speed_calc)[0,1])
# correlation between running_time and adjusted_running_time
print('corr running_time with adjusted_running_time:', np.corrcoef(df['running_time'], df['adjusted_running_time'])[0,1])
# maybe running_time is seconds? Compare with adjusted running time in ms
print('corr running_time with adjusted_running_time (ms) inverse?')
print(np.corrcoef(df['running_time'], 1/df['adjusted_running_time'])[0,1])

