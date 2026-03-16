import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
for col in ['adjusted_running_time','age','gender']:
    print(col, df['running_time'].corr(df[col]))
