import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
num_words = df['num_words']
adj_time = df['age']
wpm_adj = num_words * 60000 / adj_time
ratio = df['running_time'] / wpm_adj
print(ratio.describe())
print('median ratio', ratio.median())
