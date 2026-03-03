import pandas as pd

df = pd.read_csv('amtl.csv')

def constant_per_specimen(col):
    return df.groupby('prob_male')[col].nunique().max()

for col in ['pop','num_amtl','stdev_age','genus','age']:
    print(col, 'max nunique per specimen', df.groupby('prob_male')[col].nunique().max())

# show for a random specimen
spec = df['prob_male'].iloc[0]
print('specimen', spec)
print(df[df['prob_male']==spec])
