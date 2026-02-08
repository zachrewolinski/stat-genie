import pandas as pd

df = pd.read_csv('amtl.csv')
df = df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus',
    'genus': 'num_amtl',
    'age': 'num_sockets',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'prob_male': 'specimen_id',
    'specimen': 'population'
})
print(df.columns.tolist())
print(df.columns[df.columns.duplicated()].tolist())
