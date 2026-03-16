import pandas as pd

df = pd.read_csv('amtl.csv')
print(sorted(df['stdev_age'].unique()))
