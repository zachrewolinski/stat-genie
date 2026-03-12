import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    print('\n', col)
    print(df.groupby('sockets')[col].agg(['mean','min','max','std']))
