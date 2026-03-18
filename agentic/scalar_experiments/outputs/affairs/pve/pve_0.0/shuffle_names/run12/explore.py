import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

# correlation with row index
idx = np.arange(len(df))
for col in ['education','age']:
    if df[col].dtype != 'object':
        corr = np.corrcoef(idx, df[col])[0,1]
        print(col, 'corr with index', corr)

# check if education is unique and if close to index scaled
print('education unique', df['education'].nunique())

# basic stats for education
print('education stats', df['education'].describe())

# check if education equals rounded something of age or others
for col in ['age','occupation','children','rating','yearsmarried','rownames','affairs']:
    if df[col].dtype != 'object':
        corr = np.corrcoef(df['education'], df[col])[0,1]
        print('corr education with', col, corr)

# check if age close to any known discrete values
orig_vals = np.array([0,1,2,3,7,12])
# find nearest original value after rounding to nearest integer
age = df['age'].values
nearest = orig_vals[np.argmin(np.abs(age[:,None]-orig_vals[None,:]), axis=1)]
print('age nearest to orig mean abs diff', np.mean(np.abs(age-nearest)))
print('age min/max', age.min(), age.max())

# Check if age values are within [-0.5,12.5], and count negatives
print('age negatives', np.sum(age<0), 'out of', len(age))

# check if age resembles centered affairs by adding constant 4?
for offset in [0,1,2,3,4,5,6,7,8]:
    nearest = orig_vals[np.argmin(np.abs((age+offset)[:,None]-orig_vals[None,:]), axis=1)]
    mean_abs = np.mean(np.abs((age+offset)-nearest))
    print('offset', offset, 'mean abs', mean_abs)
