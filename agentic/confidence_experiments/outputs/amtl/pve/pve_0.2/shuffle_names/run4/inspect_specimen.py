import pandas as pd

df=pd.read_csv('amtl.csv')
# pick first specimen
spec=df['prob_male'].iloc[0]
sub=df[df['prob_male']==spec]
print(sub)

# check within-specimen std for numeric columns
cols=['genus','age','pop','num_amtl','stdev_age']
print('\nWithin-specimen std (first 5 specimens):')
for spec in df['prob_male'].unique()[:5]:
    sub=df[df['prob_male']==spec]
    stds=sub[cols].std()
    print(spec)
    print(stds)
