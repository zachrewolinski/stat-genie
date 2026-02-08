import pandas as pd

df = pd.read_csv('amtl.csv')
for col in ['num_amtl','pop','age','genus']:
    print('\n', col, 'by sockets max/min')
    print(df.groupby('sockets')[col].agg(['min','max','mean']).round(3))

# check if num_amtl correlates with pop (age) by genus
print('\nnum_amtl by tooth_class genus mean')
print(df.groupby('tooth_class')['num_amtl'].mean().sort_values())
print('\nnum_amtl by sockets mean')
print(df.groupby('sockets')['num_amtl'].mean())

# check if genus column could be num_amtl counts: compute mean by genus
print('\n genus col by tooth_class mean')
print(df.groupby('tooth_class')['genus'].mean())

# check if age column seems counts by sockets
print('\n age col by sockets mean')
print(df.groupby('sockets')['age'].mean())

