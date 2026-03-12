import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)

# test relationships between adjusted_running_time, age, gender
# We'll see which column equals sum of other two approx
cols = ['adjusted_running_time','age','gender']

# compute residuals for possibilities
# assume running_time = adjusted + scroll
for run_col in cols:
    for adj_col in cols:
        for scroll_col in cols:
            if len({run_col, adj_col, scroll_col}) < 3:
                continue
            residual = df[run_col] - (df[adj_col] + df[scroll_col])
            print(f"{run_col} ?= {adj_col}+{scroll_col} -> residual mean {residual.mean():.2f} std {residual.std():.2f} median {residual.median():.2f}")

