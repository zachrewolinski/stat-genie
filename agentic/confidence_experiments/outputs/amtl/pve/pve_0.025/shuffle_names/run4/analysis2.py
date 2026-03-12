import pandas as pd


df=pd.read_csv('amtl.csv')
print('pop mean by genus (tooth_class):')
print(df.groupby('tooth_class')['pop'].mean())
print('age mean by genus (tooth_class):')
print(df.groupby('tooth_class')['age'].mean())
print('stdev_age mean by genus:')
print(df.groupby('tooth_class')['stdev_age'].mean())

print('num_amtl mean by genus:')
print(df.groupby('tooth_class')['num_amtl'].mean())

# check for bounds
print('age range', df['age'].min(), df['age'].max())
print('pop range', df['pop'].min(), df['pop'].max())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())

# For each genus, see relationship of pop with num_amtl
for genus in df['tooth_class'].unique():
    sub = df[df['tooth_class']==genus]
    print(genus, 'corr pop-num_amtl', sub['pop'].corr(sub['num_amtl']))

