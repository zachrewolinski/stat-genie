import pandas as pd
import json
from pathlib import Path

path = Path('crofoot.csv')

df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print('n rows', len(df))
print(df.describe(include='all'))
