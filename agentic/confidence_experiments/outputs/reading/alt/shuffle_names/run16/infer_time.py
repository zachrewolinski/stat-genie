import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate time columns
cols = ['adjusted_running_time','age','gender']

# check if any column approx equals sum of others
for a in cols:
    for b in cols:
        if a==b: continue
        for c in cols:
            if c in (a,b): continue
            diff = df[a] - (df[b] + df[c])
            print(f'{a} - ({b}+{c}) median {diff.median():.2f} mean {diff.mean():.2f} std {diff.std():.2f} min {diff.min():.2f} max {diff.max():.2f}')

# check if adjusted_running_time approx age - gender etc
for a in cols:
    for b in cols:
        if a==b: continue
        diff = df[a] - df[b]
        print(f'{a} - {b} median {diff.median():.2f} mean {diff.mean():.2f} std {diff.std():.2f}')

