import pandas as pd
import itertools
import numpy as np

df = pd.read_csv('reading.csv')
cols = ['adjusted_running_time','age','gender']

for a,b,c in itertools.permutations(cols, 3):
    diff = (df[a] - df[b] - df[c]).abs().mean()
    print(f'{a} - {b} - {c} mean abs diff: {diff:.2f}')
