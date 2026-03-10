import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print(df['running_time'].describe())
print('median', df['running_time'].median())
print('quantiles', df['running_time'].quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
