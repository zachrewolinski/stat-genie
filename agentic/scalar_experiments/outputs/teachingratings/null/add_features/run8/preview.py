import pandas as pd

pd.set_option('display.width', 120)

_df = pd.read_csv('teachingratings.csv')
print(_df.head())
print(_df.columns.tolist())
print(_df.dtypes)
print(_df.shape)
