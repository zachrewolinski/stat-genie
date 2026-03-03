import pandas as pd

df = pd.read_csv('amtl.csv')
print(df['stdev_age'].value_counts().sort_index())
print('\nBy genus:')
print(df.groupby('tooth_class')['stdev_age'].value_counts().unstack(fill_value=0))
