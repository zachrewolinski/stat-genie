import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# check if genus within 0-age
cond = (df['genus']>=0) & (df['genus']<=df['age'])
print('genus within [0, age] proportion:', cond.mean())
print('genus min/max', df['genus'].min(), df['genus'].max())
print('age min/max', df['age'].min(), df['age'].max())
# check if maybe genus is logit of proportion missing? compute logistic if num_amtl or pop etc.
for col in ['pop','num_amtl','stdev_age']:
    print(col, df[col].min(), df[col].max())
# check if any numeric column within [0,age]
for col in ['genus','pop','num_amtl','stdev_age']:
    if np.issubdtype(df[col].dtype, np.number):
        prop = ((df[col]>=0) & (df[col]<=df['age'])).mean()
        print('prop', col, prop)

# compute per tooth class mean age count
print(df.groupby('sockets')['age'].describe())

