import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
for col in ['pop','num_amtl','age']:
    vals = df[col]
    for name, func in [('log', np.log), ('sqrt', np.sqrt)]:
        v = func(vals)
        corr = np.corrcoef(v, df['genus'])[0,1]
        print(col, name, 'corr with genus', corr)
