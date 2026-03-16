import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# check within specimen id (prob_male) variation
id_col = 'prob_male'
for c in ['genus','age','pop','num_amtl','stdev_age']:
    # count groups where more than 1 unique value
    var = df.groupby(id_col)[c].nunique()
    print(c, 'specimens with >1 unique values:', (var>1).mean(), 'mean nunique', var.mean())

# check number of rows per specimen and sockets
print('rows per specimen mean', df.groupby(id_col).size().mean())
print('rows per specimen min/max', df.groupby(id_col).size().min(), df.groupby(id_col).size().max())

# check if sockets levels per specimen
socket_levels = df.groupby(id_col)['sockets'].nunique()
print('specimens with all 3 sockets levels:', (socket_levels==3).mean())
print(socket_levels.describe())
