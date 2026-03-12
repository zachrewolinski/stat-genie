import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

numeric_cols = [c for c in df.columns if df[c].dtype != 'object']

for col in ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature12','feature14','feature16','feature17','feature19']:
    if col in df:
        print(col, df['feature20'].corr(df[col]))
