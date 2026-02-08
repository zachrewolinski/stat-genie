import pandas as pd

df = pd.read_csv('amtl.csv')

# inspect distributions by genus
for col in ['pop','num_amtl','age','genus','stdev_age']:
    print('\n', col)
    print(df.groupby('tooth_class')[col].describe()[['min','mean','max']])

# check if stdev_age values by species maybe sex distribution? 
print('\nprob_male (stdev_age) value counts')
print(df['stdev_age'].value_counts().sort_index())
