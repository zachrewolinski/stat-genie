import pandas as pd

df = pd.read_csv('amtl.csv')

for col in ['age','pop','num_amtl','genus','stdev_age']:
    print('\nMean of', col, 'by sockets:')
    print(df.groupby('sockets')[col].mean())

print('\nMean of numeric by tooth_class (genus):')
for col in ['age','pop','num_amtl','genus','stdev_age']:
    print(col)
    print(df.groupby('tooth_class')[col].mean())

