import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

# Check if each speed has all scrolling_time values
vals = df.groupby('speed')['scrolling_time'].nunique()
print(vals.value_counts())

# Check if each speed has 6 rows
counts = df['speed'].value_counts()
print('speed count unique values', counts.nunique())
print(counts.head())

# For a sample speed, list scrolling_time
sample = df['speed'].iloc[0]
print('sample speed', sample)
print(df[df['speed']==sample]['scrolling_time'].tolist())
