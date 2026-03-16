import pandas as pd

df = pd.read_csv('amtl.csv')
print(df['stdev_age'].value_counts().sort_index())
