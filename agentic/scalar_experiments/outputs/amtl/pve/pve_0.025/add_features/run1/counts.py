import pandas as pd

df = pd.read_csv('amtl.csv')
print(df['genus'].value_counts())
