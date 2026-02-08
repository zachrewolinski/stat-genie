import pandas as pd

df = pd.read_csv('affairs.csv')
print('age unique:', sorted(df['age'].unique()))
print('affairs unique:', sorted(df['affairs'].unique()))
print('rating unique:', sorted(df['rating'].unique()))
print('children unique sample:', sorted(df['children'].unique())[:20])
print('religiousness unique:', df['religiousness'].unique())
