import pandas as pd
import json

csv_path = 'teachingratings.csv'

df = pd.read_csv(csv_path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# show unique counts for object columns
obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    print('\n', col, df[col].value_counts(dropna=False))
