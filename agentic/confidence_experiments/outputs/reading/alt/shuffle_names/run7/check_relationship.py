import pandas as pd
import numpy as np

df=pd.read_csv('reading.csv')

# Check if adjusted_running_time == age + gender
sum_diff = df['adjusted_running_time'] - (df['age'] + df['gender'])
print('adjusted_running_time - (age+gender) min', sum_diff.min(), 'max', sum_diff.max(), 'mean', sum_diff.mean())
print('num nonzero', (sum_diff!=0).sum())
print('sample nonzero', sum_diff[sum_diff!=0].head())

# Check if adjusted_running_time - age equals gender
other_diff = df['adjusted_running_time'] - df['age'] - df['gender']
print('same as above', other_diff.min(), other_diff.max())

# Check if age == adjusted_running_time - gender
age_diff = df['age'] - (df['adjusted_running_time'] - df['gender'])
print('age - (adjusted_running_time - gender) min', age_diff.min(), 'max', age_diff.max(), 'mean', age_diff.mean())
print('nonzero', (age_diff!=0).sum())
