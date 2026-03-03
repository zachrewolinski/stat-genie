import pandas as pd

df=pd.read_csv('amtl.csv')
# select specimen
spec = df['prob_male'].iloc[0]
print('spec', spec)
print(df[df['prob_male']==spec])

# check if num_amtl varies by sockets within specimen
sample = df[df['prob_male']==spec]
print('num_amtl', sample['num_amtl'].tolist())
print('age', sample['age'].tolist())
print('pop', sample['pop'].tolist())
print('genus', sample['genus'].tolist())
print('stdev_age', sample['stdev_age'].tolist())
