import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
k = df['num_words'] * 60000 / (df['adjusted_running_time'] * df['running_time'])
print(k.describe())
print(k.head())
