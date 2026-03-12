import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
prop = df['genus'] / df['age']
print(prop.describe())
print('min', prop.min(), 'max', prop.max())
print('fraction outside [0,1]:', ((prop<0)|(prop>1)).mean())

