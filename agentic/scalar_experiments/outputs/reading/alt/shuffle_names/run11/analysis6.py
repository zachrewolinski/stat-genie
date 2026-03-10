import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

wpm_adj = df['num_words'] / df['adjusted_running_time'] * 60000
ratio = df['running_time'] / wpm_adj
print(ratio.describe())
print('median ratio', ratio.median())
