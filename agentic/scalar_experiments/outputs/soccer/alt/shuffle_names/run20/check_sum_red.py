import pandas as pd


df = pd.read_csv('soccer.csv')

sum_red = df['meanExp'] + df['yellowCards']

for col in df.columns:
    if col in ['meanExp','yellowCards']:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        if (df[col] == sum_red).all():
            print('col equals sum', col)
