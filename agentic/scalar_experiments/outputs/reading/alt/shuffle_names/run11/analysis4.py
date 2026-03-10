import pandas as pd

path='reading.csv'
df=pd.read_csv(path)
print(df['language'].value_counts())
