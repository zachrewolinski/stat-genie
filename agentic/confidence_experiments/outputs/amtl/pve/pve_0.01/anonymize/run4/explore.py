import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
num = df.select_dtypes(include=[np.number])
print(num.describe().T)
print('feature3 min max', df['feature3'].min(), df['feature3'].max())
print('feature3 unique approx', df['feature3'].nunique())
print('feature4 unique', df['feature4'].unique()[:20])
print('feature3 head', df['feature3'].head(10).tolist())
print('feature4 head', df['feature4'].head(10).tolist())

# check if feature3 maybe standardized counts: correlation with feature4?
print('corr feature3 feature4', df['feature3'].corr(df['feature4']))
# check if feature3 per specimen? maybe per class
print(df.groupby('feature8')['feature3'].describe())
print(df.groupby('feature8')['feature4'].describe())
