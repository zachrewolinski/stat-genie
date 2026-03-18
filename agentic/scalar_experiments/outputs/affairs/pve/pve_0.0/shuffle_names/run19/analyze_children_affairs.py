import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
children_col = 'religiousness'  # yes/no

print('Children col value counts:', df[children_col].value_counts())

# compute mean differences for numeric columns
for col in df.columns:
    if col == children_col or df[col].dtype == 'object':
        continue
    grp = df.groupby(children_col)[col].agg(['mean','median','std','count'])
    diff = grp.loc['yes','mean'] - grp.loc['no','mean']
    print('\n', col)
    print(grp)
    print('mean difference (yes - no):', diff)

# Also check affairs category (affairs column) with children
print('\nCrosstab for affairs by children:')
print(pd.crosstab(df['affairs'], df[children_col], normalize='columns'))
