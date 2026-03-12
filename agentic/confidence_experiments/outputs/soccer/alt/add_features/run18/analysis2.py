import pandas as pd
import numpy as np

_df = pd.read_csv('soccer.csv')
_df['skin'] = _df[['rater1','rater2']].mean(axis=1)
print(_df['skin'].describe())
print(_df['skin'].value_counts(dropna=False).sort_index())
