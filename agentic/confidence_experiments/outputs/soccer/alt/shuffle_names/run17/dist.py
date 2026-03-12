import os, sys
cwd=os.getcwd(); sys.path=[p for p in sys.path if p not in ("", cwd)]
import pandas as pd
import numpy as np

cols=['player','yellowReds','meanExp','yellowCards']
df=pd.read_csv('soccer.csv')
for c in cols:
    s=df[c]
    print(c, 'mean', s.mean(), 'max', s.max(), 'zero_pct', (s==0).mean())
    print('value_counts head', s.value_counts().head())

