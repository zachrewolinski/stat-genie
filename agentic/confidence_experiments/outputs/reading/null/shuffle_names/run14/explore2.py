import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

df = pd.read_csv('reading.csv')

for col in df.columns:
    ser = df[col]
    nunique = ser.nunique(dropna=False)
    print(f"\n{col} | dtype={ser.dtype} | nunique={nunique}")
    if ser.dtype == 'object':
        vals = ser.dropna().unique()[:10]
        print(" sample:", vals)
    else:
        print(" min", ser.min(), "max", ser.max(), "mean", ser.mean())
