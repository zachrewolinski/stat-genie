import pandas as pd

df = pd.read_csv('amtl.csv')

for col in ['genus','num_amtl']:
    print('\n', col)
    print(df.groupby('tooth_class')[col].mean())
