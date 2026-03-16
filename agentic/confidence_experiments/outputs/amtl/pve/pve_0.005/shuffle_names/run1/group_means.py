import pandas as pd
_df = pd.read_csv('amtl.csv')
# group by genus (tooth_class column) for each numeric variable
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print('\n', col)
    print(_df.groupby('tooth_class')[col].mean().sort_values())
