import pandas as pd

_df = pd.read_csv('amtl.csv')
print(_df.describe(include='all'))
