import pandas as pd

df = pd.read_csv('amtl.csv')

print('max genus > age?', (df['genus'] > df['age']).sum())
print('max ratio', (df['genus']/df['age']).max())
print('min age', df['age'].min(), 'max age', df['age'].max())

# count rows where age is 0
print('age zero', (df['age']==0).sum())
