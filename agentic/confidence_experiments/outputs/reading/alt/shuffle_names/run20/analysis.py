import pandas as pd
import json
from pathlib import Path

path = Path('reading.csv')
print('reading', path)
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
