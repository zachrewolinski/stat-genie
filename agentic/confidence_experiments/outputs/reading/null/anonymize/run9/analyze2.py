import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv('reading.csv')
print(df.head(5))
print('feature4 first5', df['feature4'].head().tolist())
print('feature5 first5', df['feature5'].head().tolist())
print('feature20 first5', df['feature20'].head().tolist())
