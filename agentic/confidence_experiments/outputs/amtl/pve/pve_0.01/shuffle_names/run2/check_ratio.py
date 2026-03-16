import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

ratio = df['genus'] / df['age']
print('ratio min', ratio.min(), 'max', ratio.max())
print('ratio pct within 0-1', ((ratio>=0)&(ratio<=1)).mean())
print('genus <= age pct', (df['genus'] <= df['age']).mean())
print('genus >=0 pct', (df['genus']>=0).mean())

# check for any rows where genus is negative or greater than age
print('neg genus count', (df['genus']<0).sum())
print('genus > age count', (df['genus']>df['age']).sum())
