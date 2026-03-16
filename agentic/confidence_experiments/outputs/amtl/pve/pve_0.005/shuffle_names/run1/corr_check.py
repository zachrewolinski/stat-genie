import pandas as pd
import numpy as np
_df = pd.read_csv('amtl.csv')
for col in ['genus','num_amtl','pop','stdev_age']:
    corr = _df[col].corr(_df['age'])
    print('corr with age', col, corr)
# with sockets class? not numeric.
# Compare genus by tooth_class (genus?), etc
print(_df.groupby('sockets')['genus'].mean())
print(_df.groupby('sockets')['num_amtl'].mean())
