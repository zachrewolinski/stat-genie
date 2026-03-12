import pandas as pd
import numpy as np


df = pd.read_csv('crofoot.csv')
print('rows', len(df))
print(df.head())

# summary unique counts
summary = []
for col in df.columns:
    summary.append((col, df[col].min(), df[col].max(), df[col].nunique()))
print('summary:')
for s in summary:
    print(s)

