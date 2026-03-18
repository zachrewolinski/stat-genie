import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
num_cols = df.select_dtypes(include=[np.number]).columns
qs = [0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]
for col in num_cols:
    print("\n",col)
    print(df[col].quantile(qs))
