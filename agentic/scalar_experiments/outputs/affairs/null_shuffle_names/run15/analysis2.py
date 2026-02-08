import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df['age'].value_counts().sort_index())
print(df['affairs'].value_counts().sort_index())
print(df['rating'].value_counts().sort_index())
print(df['religiousness'].value_counts())
