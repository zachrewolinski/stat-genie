import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# which combination yields best? check correlations
for total_col in ['adjusted_running_time','age']:
    for base_col in ['age','adjusted_running_time']:
        if total_col==base_col:
            continue
        diff = df[total_col] - df[base_col]
        corr = diff.corr(df['gender'])
        print(f'corr ({total_col}-{base_col}) vs gender:', corr)
        print(f'mean diff {diff.mean()}')

# check if adjusted_running_time approx age + gender
approx = df['age'] + df['gender']
print('corr adjusted_running_time vs age+gender', df['adjusted_running_time'].corr(approx))
print('mean absolute difference', (df['adjusted_running_time'] - approx).abs().mean())

# check if age approx adjusted_running_time - gender
approx2 = df['adjusted_running_time'] - df['gender']
print('corr age vs adjusted_running_time - gender', df['age'].corr(approx2))
print('mean abs diff', (df['age'] - approx2).abs().mean())
