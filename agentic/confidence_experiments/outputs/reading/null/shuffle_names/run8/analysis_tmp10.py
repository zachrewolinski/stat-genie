import pandas as pd

df = pd.read_csv('reading.csv')
print(df['language'].value_counts())
