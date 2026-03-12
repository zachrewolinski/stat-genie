import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')

for col in ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature19']:
    corr = np.corrcoef(df[col], df['feature20'])[0,1]
    print(col, corr)
