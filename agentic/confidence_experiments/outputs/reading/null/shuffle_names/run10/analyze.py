import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print('shape', df.shape)
print('columns', df.columns.tolist())
# basic stats for key cols
print(df[['language','dyslexia','dyslexia_bin','running_time','adjusted_running_time','scrolling_time','speed']].head())
