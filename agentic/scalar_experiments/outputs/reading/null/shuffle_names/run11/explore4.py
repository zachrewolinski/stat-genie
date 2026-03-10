import pandas as pd
import numpy as np

df=pd.read_csv('reading.csv')

# compare device and dyslexia_bin relationship
mask=~df['device'].isna() & ~df['dyslexia_bin'].isna()
sub=df[mask]
print('device vs dyslexia_bin crosstab')
print(pd.crosstab(sub['device'], sub['dyslexia_bin']))

# compare dyslexia and dyslexia_bin
mask=~df['dyslexia'].isna() & ~df['dyslexia_bin'].isna()
sub=df[mask]
print('\ndyslexia vs dyslexia_bin crosstab')
print(pd.crosstab(sub['dyslexia'], sub['dyslexia_bin']))

# compare device vs correct_rate
mask=~df['device'].isna() & ~df['correct_rate'].isna()
sub=df[mask]
print('\ndevice vs correct_rate crosstab')
print(pd.crosstab(sub['device'], sub['correct_rate']))

# compare dyslexia vs correct_rate
mask=~df['dyslexia'].isna() & ~df['correct_rate'].isna()
sub=df[mask]
print('\ndyslexia vs correct_rate crosstab')
print(pd.crosstab(sub['dyslexia'], sub['correct_rate']))
