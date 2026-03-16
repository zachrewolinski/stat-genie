import pandas as pd

df = pd.read_csv('amtl.csv')

# Count rows where num_amtl > age
print('num_amtl > age:', (df['num_amtl'] > df['age']).sum())
print('num_amtl > pop:', (df['num_amtl'] > df['pop']).sum())
print('age > pop:', (df['age'] > df['pop']).sum())

# show some rows with large num_amtl
print(df[['num_amtl','age','pop']].sort_values('num_amtl', ascending=False).head(10))

# check if genus maybe counts? ensure if genus positive etc
print('genus min/max', df['genus'].min(), df['genus'].max())

# check if num_amtl and pop are integers
print('num_amtl integer-like', ((df['num_amtl'] % 1) == 0).mean())
print('pop integer-like', ((df['pop'] % 1) == 0).mean())
print('age integer-like', ((df['age'] % 1) == 0).mean())

# maybe num_amtl is sockets? check if num_amtl around age? compute ratio
ratio = df['num_amtl'] / df['age']
print('ratio summary', ratio.min(), ratio.max(), ratio.mean())

# If pop is age, check correlation with num_amtl maybe doesn't make sense
print('corr pop vs num_amtl', df['pop'].corr(df['num_amtl']))

