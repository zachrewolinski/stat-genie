import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')
print('shape', _df.shape)
print(_df.head())
print(_df.dtypes)
print(_df.describe(include='all').transpose().head(15))

# Check unique values for categorical
for col in ['feature1','feature8']:
    print(col, _df[col].unique())

# Check ranges for feature3,4
print('feature3 min/max', _df['feature3'].min(), _df['feature3'].max())
print('feature4 min/max', _df['feature4'].min(), _df['feature4'].max())
print('feature5 min/max', _df['feature5'].min(), _df['feature5'].max())
print('feature7 unique', sorted(_df['feature7'].unique()))

