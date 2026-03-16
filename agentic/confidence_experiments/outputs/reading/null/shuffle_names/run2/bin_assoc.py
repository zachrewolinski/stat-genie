import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for bin_col in ['dyslexia_bin','correct_rate']:
    print('\n', bin_col)
    for cat_col in ['device','dyslexia']:
        corr = np.corrcoef(df[bin_col], df[cat_col])[0,1]
        print('corr with', cat_col, corr)
        ct = pd.crosstab(df[cat_col], df[bin_col], normalize='index')
        print(ct)
