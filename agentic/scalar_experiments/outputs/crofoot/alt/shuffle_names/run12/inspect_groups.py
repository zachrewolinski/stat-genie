import pandas as pd
import numpy as np


df = pd.read_csv('crofoot.csv')

# candidate group id columns
id_cols = ['n_other', 'dist_other']

for id_col in id_cols:
    print('\nID col:', id_col)
    for col in df.columns:
        if col == id_col:
            continue
        # check if col values constant within id
        grouped = df.groupby(id_col)[col].nunique()
        max_unique = grouped.max()
        if max_unique == 1:
            print(f'  constant by {id_col}: {col}')

