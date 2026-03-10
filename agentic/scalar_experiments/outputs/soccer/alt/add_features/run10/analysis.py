import pandas as pd

csv_path = 'soccer.csv'

df = pd.read_csv(csv_path)
print(df.shape)
print(df.columns.tolist())
print(df.head())
