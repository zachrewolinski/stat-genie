import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

for col in ['self_employed','accept','deny']:
    print(col, df[col].mean(), df[col].value_counts().to_dict())

print('corr self_employed vs deny', df['self_employed'].corr(df['deny']))
print('corr self_employed vs accept', df['self_employed'].corr(df['accept']))
print('corr accept vs deny', df['accept'].corr(df['deny']))

