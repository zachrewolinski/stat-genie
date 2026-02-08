import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())

# Check if genus <= age (if genus is num_missing and age is sockets)
print('genus<=age fraction', (df['genus'] <= df['age']).mean())
print('genus max', df['genus'].max(), 'age max', df['age'].max())

# Check if num_amtl <= age or genus
print('num_amtl<=age fraction', (df['num_amtl'] <= df['age']).mean())
print('num_amtl<=genus fraction', (df['num_amtl'] <= df['genus']).mean())

# Check for integer-like columns
for col in ['genus','age','pop','num_amtl','stdev_age']:
    series = df[col]
    frac_int = np.mean(np.isclose(series, np.round(series)))
    print(col, 'frac_int', frac_int, 'min', series.min(), 'max', series.max())

# Group by tooth_class (genus)
print(df['tooth_class'].value_counts())

# Check stdev_age unique values
print('stdev_age unique', sorted(df['stdev_age'].unique())[:10], 'count', df['stdev_age'].nunique())

