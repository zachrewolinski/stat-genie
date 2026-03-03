import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# Compare num_amtl to pop
print('num_amtl <= pop:', (df['num_amtl'] <= df['pop']).mean())
print('num_amtl > pop count', (df['num_amtl'] > df['pop']).sum())

# Compare age to num_amtl or pop
for col in ['genus','age','pop','num_amtl']:
    print(col, 'min', df[col].min(), 'max', df[col].max())

# Check if age correlates strongly with num_amtl or pop
print('corr age-pop', df[['age','pop']].corr().iloc[0,1])
print('corr age-num_amtl', df[['age','num_amtl']].corr().iloc[0,1])

# Check if any column looks like count of sockets: integer and maybe > num_amtl? use rounding
for col in ['genus','age','pop','num_amtl']:
    int_like = (df[col].round().astype(int) == df[col].round()).mean()
    print(col, 'rounded integer ratio', int_like)

# if age is count of sockets, should be >= num_amtl maybe? test
print('age >= num_amtl', (df['age'] >= df['num_amtl']).mean())
print('age >= pop', (df['age'] >= df['pop']).mean())

