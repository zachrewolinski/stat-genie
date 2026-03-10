import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
for time_col in ['feature4','feature5']:
    wpm = df['feature7'] / (df[time_col]/60000.0)
    print(time_col, 'wpm summary', wpm.describe())

