import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('amtl.csv')

# Rename for convenience

genus_col = 'tooth_class'  # values of genus

# summary stats by genus
summary = df.groupby(genus_col)[['genus','age','pop','num_amtl','stdev_age']].agg(['mean','std','min','max'])
print(summary)

# summary by tooth class (sockets column)
summary2 = df.groupby('sockets')[['genus','age','pop','num_amtl','stdev_age']].agg(['mean','std','min','max'])
print('\nBy sockets (tooth class values):')
print(summary2)

# check unique values counts for numeric columns
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print(col, 'unique', df[col].nunique())

