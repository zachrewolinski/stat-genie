import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('mean', df['feature3'].mean())
print('std', df['feature3'].std())
print('min', df['feature3'].min())
print('max', df['feature3'].max())
