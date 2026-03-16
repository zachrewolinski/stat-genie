import pandas as pd

df = pd.read_csv('reading.csv')

# check retake_trial vs scrolling_time combos
print('unique retake_trial values', sorted(df['retake_trial'].unique()))
print('unique scrolling_time', sorted(df['scrolling_time'].unique()))

print('retake_trial by scrolling_time:')
print(df.groupby('scrolling_time')['retake_trial'].nunique())
print(df.groupby('scrolling_time')['retake_trial'].unique())

print('num_words by scrolling_time nunique')
print(df.groupby('scrolling_time')['num_words'].nunique())

print('num_words unique count', df['num_words'].nunique())

# check if num_words values align with retake_trial values maybe via scaling
print('retake_trial summary', df['retake_trial'].describe())
print('num_words summary', df['num_words'].describe())

