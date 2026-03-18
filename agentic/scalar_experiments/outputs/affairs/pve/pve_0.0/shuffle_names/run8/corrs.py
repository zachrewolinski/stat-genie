import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

print('corr education-age', df['education'].corr(df['age']))
print('corr education-occupation', df['education'].corr(df['occupation']))
print('corr education-children', df['education'].corr(df['children']))
