import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
wpm5 = df['feature7'] / (df['feature5']/60000.0)
ratio = df['feature20'] / wpm5
print('ratio summary', ratio.describe())
print(pd.DataFrame({'feature7':df['feature7'], 'feature5':df['feature5'], 'wpm5':wpm5, 'feature20':df['feature20'], 'ratio':ratio}).head(10))
