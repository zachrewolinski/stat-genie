import pandas as pd

df=pd.read_csv('affairs.csv')
print(df['education'].sort_values().head(20).to_list())
print(df['education'].sort_values().tail(20).to_list())
print(df['education'].nunique())
print(df['education'].is_monotonic_increasing)
print(df['education'].corr(df.index))
