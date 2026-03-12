import pandas as pd

df = pd.read_csv('reading.csv')
print('feature4 mean', df['feature4'].mean())
print('feature5 mean', df['feature5'].mean())
print('feature4 max', df['feature4'].max())
print('feature5 max', df['feature5'].max())
