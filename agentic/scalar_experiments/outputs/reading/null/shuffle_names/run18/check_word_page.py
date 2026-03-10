import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

print(pd.crosstab(df['scrolling_time'], df['retake_trial']))
print('\nnum_words by scrolling_time:')
print(df.groupby('scrolling_time')['num_words'].unique())
