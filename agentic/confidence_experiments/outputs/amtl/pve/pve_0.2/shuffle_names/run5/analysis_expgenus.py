import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
expgen = np.exp(df['genus'])
print('expgen range', expgen.min(), expgen.max(), expgen.mean())
print('corr expgen with num_amtl', np.corrcoef(expgen, df['num_amtl'])[0,1])
print('corr expgen with pop', np.corrcoef(expgen, df['pop'])[0,1])
