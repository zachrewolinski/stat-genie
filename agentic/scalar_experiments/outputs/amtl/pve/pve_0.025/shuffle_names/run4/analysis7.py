import pandas as pd
import numpy as np


df=pd.read_csv('amtl.csv')
# check if genus or num_amtl values are close to integer increments like 0.25
for col in ['genus','num_amtl','pop']:
    vals = df[col].values
    # check closeness to quarters
    frac_quarter = np.mean(np.isclose(vals*4, np.round(vals*4)))
    frac_tenth = np.mean(np.isclose(vals*10, np.round(vals*10)))
    print(col, 'frac quarter', frac_quarter, 'frac tenth', frac_tenth)

# show smallest and largest values of genus
print('genus smallest 10', np.sort(df['genus'].values)[:10])
print('genus largest 10', np.sort(df['genus'].values)[-10:])

# check if genus correlates with age or sockets counts
print('corr genus-age', df['genus'].corr(df['age']))
print('corr genus-pop', df['genus'].corr(df['pop']))

