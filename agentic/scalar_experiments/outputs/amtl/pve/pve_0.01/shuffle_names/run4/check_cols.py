import pandas as pd

df=pd.read_csv('amtl.csv')
df = df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'age': 'num_sockets',
    'genus': 'amtl_freq'
})
print(df.columns)
print('duplicates', list(df.columns[df.columns.duplicated()]))
print('genus col type', type(df['genus']))
print(df['genus'].head())
