import pandas as pd

df = pd.read_csv('amtl.csv')

# check if genus <= age
print('genus <= age proportion', (df['genus'] <= df['age']).mean())
print('num_amtl <= age proportion', (df['num_amtl'] <= df['age']).mean())

# check if genus <= pop
print('genus <= pop proportion', (df['genus'] <= df['pop']).mean())
print('num_amtl <= pop proportion', (df['num_amtl'] <= df['pop']).mean())

# check if genus is integer
print('genus unique non-int?', (df['genus'] % 1 != 0).sum())
print('age unique non-int?', (df['age'] % 1 != 0).sum())
print('num_amtl unique non-int?', (df['num_amtl'] % 1 != 0).sum())

# check possible sockets counts by tooth class
print(df.groupby('sockets')[['genus','age','num_amtl']].agg(['min','max','mean']).round(2))
