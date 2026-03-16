import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
rate = df['genus'] / df['age']
print('rate min', rate.min(), 'max', rate.max())
print('rate share between 0 and 1', ((rate>=0)&(rate<=1)).mean())

