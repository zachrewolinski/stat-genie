import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
rate = df['genus'] / df['age']
print(rate.describe())
print('rate min/max', rate.min(), rate.max())
