import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
ratio = df['adjusted_running_time'] / df['running_time']
print(ratio.describe())
ratio2 = df['age'] / df['running_time']
print('age/running_time', ratio2.describe())
