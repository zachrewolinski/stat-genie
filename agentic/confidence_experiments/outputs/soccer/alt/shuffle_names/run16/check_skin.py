import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')
print('corr rater1 vs nExp', df['rater1'].corr(df['nExp']))
print('unique rater1', sorted(df['rater1'].unique()))
print('unique nExp', sorted(df['nExp'].unique()))
