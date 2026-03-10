import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
ratio_adj = df['adjusted_running_time'] / df['running_time']
ratio_age = df['age'] / df['running_time']
print('ratio adjusted_running_time / running_time stats')
print(ratio_adj.describe())
print('ratio age / running_time stats')
print(ratio_age.describe())
