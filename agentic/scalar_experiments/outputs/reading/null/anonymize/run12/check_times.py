import pandas as pd

df = pd.read_csv('reading.csv')

f4 = df['feature4']
f5 = df['feature5']

print('feature5 <= feature4 proportion:', (f5 <= f4).mean())
print('feature4 <= feature5 proportion:', (f4 <= f5).mean())
print('median feature4', f4.median(), 'median feature5', f5.median())
