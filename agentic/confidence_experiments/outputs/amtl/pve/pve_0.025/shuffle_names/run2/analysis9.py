import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
frac = np.abs(df['genus'] - np.round(df['genus']))
print('genus within 0.1 of integer', (frac<0.1).mean())
print('genus within 0.25 of integer', (frac<0.25).mean())
