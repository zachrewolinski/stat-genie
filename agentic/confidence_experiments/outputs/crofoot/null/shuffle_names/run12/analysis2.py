import pandas as pd
import numpy as np

df=pd.read_csv('crofoot.csv')
for col in df.columns:
    print(col, df[col].nunique(), df[col].min(), df[col].max(), sorted(df[col].unique())[:10])
