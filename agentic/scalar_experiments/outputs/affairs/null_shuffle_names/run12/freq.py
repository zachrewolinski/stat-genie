import pandas as pd

_df = pd.read_csv('affairs.csv')
print(_df['age'].value_counts().sort_index())
print(_df['religiousness'].value_counts())
