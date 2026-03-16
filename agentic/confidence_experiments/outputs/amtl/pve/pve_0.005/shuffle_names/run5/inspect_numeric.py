import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
num_cols = df.select_dtypes(include='number').columns
print(num_cols)
for c in num_cols:
    print(c, df[c].min(), df[c].max(), df[c].mean(), df[c].std())
