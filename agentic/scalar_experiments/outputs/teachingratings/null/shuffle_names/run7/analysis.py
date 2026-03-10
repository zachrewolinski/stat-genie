import pandas as pd

path = 'teachingratings.csv'

df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print('\ninfo')
print(df.dtypes)
print('\nmissing', df.isna().sum())
print('\nsummary beauty')
print(df['beauty'].describe())
# identify rating column candidates
print('\nsummary allstudents')
print(df['allstudents'].describe())
