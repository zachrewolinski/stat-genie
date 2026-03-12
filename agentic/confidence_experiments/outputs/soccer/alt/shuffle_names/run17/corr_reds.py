import os, sys
cwd=os.getcwd(); sys.path=[p for p in sys.path if p not in ("", cwd)]
import pandas as pd
import numpy as np

df=pd.read_csv('soccer.csv')
print('corr meanExp vs yellowCards', df['meanExp'].corr(df['yellowCards']))
print('nonzero both', ((df['meanExp']>0) & (df['yellowCards']>0)).mean())
print('nonzero meanExp', (df['meanExp']>0).mean())
print('nonzero yellowCards', (df['yellowCards']>0).mean())

