import pandas as pd

df = pd.read_csv('amtl.csv')

for col in ['age','pop','num_amtl','stdev_age']:
    if pd.api.types.is_numeric_dtype(df[col]):
        greater = (df['genus'] > df[col]).sum()
        print(col, 'genus>col', greater, 'min col', df[col].min(), 'max col', df[col].max())
