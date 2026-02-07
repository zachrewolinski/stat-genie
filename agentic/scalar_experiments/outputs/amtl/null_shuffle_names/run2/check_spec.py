import pandas as pd

df = pd.read_csv('amtl.csv')

spec = df['prob_male'].value_counts().index[0]
print('spec', spec)
print(df[df['prob_male']==spec])

# check uniqueness per specimen
for col in ['genus','age','pop','num_amtl','stdev_age']:
    grouped = df.groupby('prob_male')[col].nunique()
    print('\n', col)
    print(grouped.value_counts().head())
