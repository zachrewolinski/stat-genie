import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

counts = df.groupby('speed')['language'].value_counts().unstack(fill_value=0)
print(counts.head())

# distribution of counts per participant
count_pairs = counts.apply(lambda row: tuple(row.values), axis=1).value_counts().head(10)
print('\nTop count patterns:')
print(count_pairs)
