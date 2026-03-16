import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path('amtl.csv')

df = pd.read_csv(csv_path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
