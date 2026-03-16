import pandas as pd

_df = pd.read_csv('panda_nuts.csv')
print(_df['sex'].value_counts().to_string())
print('\n')
print(_df['help'].value_counts().to_string())
