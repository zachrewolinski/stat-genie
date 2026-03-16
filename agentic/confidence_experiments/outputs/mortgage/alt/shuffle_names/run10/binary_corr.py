import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

binary_cols = ['denied_PMI', 'consumer_credit', 'accept', 'loan_to_value', 'PI_ratio', 'self_employed', 'female', 'deny']

for i, c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        corr = df[[c1,c2]].corr().iloc[0,1]
        if abs(corr) > 0.9:
            print(c1, c2, corr)
