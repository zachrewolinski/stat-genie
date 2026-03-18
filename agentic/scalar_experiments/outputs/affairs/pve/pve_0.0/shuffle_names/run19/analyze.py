import pandas as pd
import json

# Load data
csv_path = 'affairs.csv'

df = pd.read_csv(csv_path)

print('Columns:', df.columns.tolist())
print('Head:')
print(df.head())
print('Describe:')
print(df.describe(include='all'))

# Look at unique values for potential categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        print('\nCol', col, 'unique values:', df[col].unique()[:10])
    else:
        # integer small range
        if df[col].nunique() <= 10:
            print('\nCol', col, 'unique values:', sorted(df[col].unique()))
