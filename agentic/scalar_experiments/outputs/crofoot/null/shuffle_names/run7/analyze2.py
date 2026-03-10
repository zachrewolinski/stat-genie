import pandas as pd
import numpy as np

df = pd.read_csv('crofoot.csv')
for col in df.columns:
    print(col, 'min', df[col].min(), 'max', df[col].max(), 'unique', sorted(df[col].unique())[:10], 'nunique', df[col].nunique())
