import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
edu = df['education']
# distance to nearest 1000
nearest = ((edu/1000).round()*1000)
print('mean abs dist to 1000 multiple', (edu-nearest).abs().mean())
print('median abs dist', (edu-nearest).abs().median())
print('min abs dist', (edu-nearest).abs().min())
print('max abs dist', (edu-nearest).abs().max())
# count within 50 of 1000 multiple
print('within 50 of 1000 multiple', ((edu-nearest).abs()<=50).mean())
# show some values near 1000 multiples
print(edu[(edu-nearest).abs()<=50].head(10))
