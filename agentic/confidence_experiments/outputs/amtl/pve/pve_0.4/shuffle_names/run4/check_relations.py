import pandas as pd

df = pd.read_csv('amtl.csv')

# Check if num_amtl <= age (if age = sockets)
print('num_amtl <= age count:', (df['num_amtl'] <= df['age']).sum(), 'of', len(df))
print('num_amtl > age max diff:', (df['num_amtl'] - df['age']).max())

# Check if genus <= age
print('genus <= age count:', (df['genus'] <= df['age']).sum(), 'of', len(df))
print('genus > age max diff:', (df['genus'] - df['age']).max())

# Maybe age <= num_amtl? 
print('age <= num_amtl count:', (df['age'] <= df['num_amtl']).sum(), 'of', len(df))

# maybe age could be missing count? Are age values integer 2-14; sockets? check if age <= 14 typical sockets for class? 

# Check if num_amtl relates to sockets categories
print('num_amtl by sockets mean:', df.groupby('sockets')['num_amtl'].mean())
print('genus by sockets mean:', df.groupby('sockets')['genus'].mean())

# Check if num_amtl less than some plausible socket count by class if we map from sockets categories
# We will compute per row if num_amtl <= age

# Check ranges by tooth_class (genus)
print('num_amtl by tooth_class mean:', df.groupby('tooth_class')['num_amtl'].mean())
print('genus by tooth_class mean:', df.groupby('tooth_class')['genus'].mean())

# Check if age correlates with pop (age?)
print('corr age-pop', df['age'].corr(df['pop']))
print('corr age-num_amtl', df['age'].corr(df['num_amtl']))
print('corr pop-num_amtl', df['pop'].corr(df['num_amtl']))
print('corr pop-genus', df['pop'].corr(df['genus']))
