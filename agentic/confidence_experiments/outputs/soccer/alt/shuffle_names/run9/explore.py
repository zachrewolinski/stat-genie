import pandas as pd

pd.set_option('display.max_columns', None)

# Load dataset
csv_path = 'soccer.csv'
df = pd.read_csv(csv_path)

print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head(3))
print('\nDtypes:')
print(df.dtypes)

print('\nNumeric describe:')
print(df.describe().transpose())

print('\nMissing values:')
print(df.isna().sum())
