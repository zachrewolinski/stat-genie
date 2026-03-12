import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

cols = df.columns

rows = []
for col in cols:
    nunique = df.groupby('speed')[col].nunique(dropna=True)
    # percentage of participants with constant value
    pct_const = (nunique==1).mean()*100
    pct_two = (nunique==2).mean()*100
    rows.append((col, pct_const, pct_two, nunique.value_counts().head().to_dict()))

# sort by pct_const
rows.sort(key=lambda x: -x[1])
for col, pct_const, pct_two, vc in rows:
    print(f"{col}: pct_const={pct_const:.1f}% pct_two={pct_two:.1f}% counts={vc}")
