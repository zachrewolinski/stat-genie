import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('tooth_class')['num_amtl'].describe())
