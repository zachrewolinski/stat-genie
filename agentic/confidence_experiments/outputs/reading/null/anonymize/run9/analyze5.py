import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
cols = ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature19']
for col in cols:
    corr = np.corrcoef(df[col], df['feature20'])[0,1]
    print('corr feature20 with', col, corr)
