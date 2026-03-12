import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

cols = ['feature4','feature5','feature6','feature7','feature20']

for c in cols:
    if c not in df.columns:
        continue

# correlation matrix
print(df[cols].corr())

