import pandas as pd

df = pd.read_csv('affairs.csv')
print(df['education'].sort_values().head(20).tolist())
print(df['education'].sort_values().tail(20).tolist())
