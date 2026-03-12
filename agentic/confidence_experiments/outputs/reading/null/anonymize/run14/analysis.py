import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.dtypes)

# basic stats for feature20
print('feature20 describe')
print(df['feature20'].describe())

# maybe reading speed words per minute? compute feature7 / (feature5 ms) * 60000
# to see correlation with feature20
if 'feature7' in df.columns and 'feature5' in df.columns:
    speed_calc = df['feature7'] / (df['feature5'] / 60000.0)
    print('speed_calc describe')
    print(speed_calc.describe())
    corr = np.corrcoef(speed_calc.fillna(0), df['feature20'].fillna(0))[0,1]
    print('corr speed_calc vs feature20', corr)

# check correlation between feature20 and reading time features
for col in ['feature4','feature5','feature6','feature7']:
    if col in df.columns:
        print('corr feature20 vs', col, np.corrcoef(df['feature20'], df[col])[0,1])
