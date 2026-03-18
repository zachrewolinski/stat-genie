import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

col = df['education']

# check if values are multiples of 7, 30, 365
for base in [7, 30, 365]:
    rem = np.mod(col, base)
    print('base', base, 'unique remainders:', np.unique(rem)[:20], 'count', len(np.unique(rem)))

# check if education values are integer
print('education integer?', np.all(np.equal(col, np.round(col))))

# show smallest values
print('smallest 20 values:', np.sort(col.values)[:20])

# check if education values correspond to e.g. 0-12 * 365 plus noise
vals = np.sort(col.values)
print('min', vals.min(), 'max', vals.max())
