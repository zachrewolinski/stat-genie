import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# try combinations for logit of ratio of two positive columns
pos_cols = ['age','pop','num_amtl']

for num in pos_cols:
    for denom in pos_cols:
        if num==denom:
            continue
        ratio = df[num]/df[denom]
        ratio = ratio.clip(1e-6, 1-1e-6)
        logit = np.log(ratio/(1-ratio))
        corr = np.corrcoef(df['genus'], logit)[0,1]
        print('corr genus vs logit({}/{})'.format(num, denom), corr)

