import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path('soccer.csv')

df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.columns)
print(df[['rater1','rater2','redCards','games']].describe())
print(df[['rater1','rater2']].isna().mean())
