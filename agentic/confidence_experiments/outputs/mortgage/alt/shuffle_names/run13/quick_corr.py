import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

# compute correlations of deny and accept with some credit variables
cols = ['bad_history','consumer_credit','mortgage_credit','loan_to_value','PI_ratio','housing_expense_ratio']
for target in ['deny','accept']:
    if target in df.columns:
        print('\n', target)
        for c in cols:
            if c in df.columns:
                corr = df[target].corr(df[c])
                print(c, corr)
