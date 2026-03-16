import pandas as pd


df=pd.read_csv('amtl.csv')

for g in df['tooth_class'].unique():
    sub = df[df['tooth_class']==g]
    print('genus', g, 'corr genus(pop?)', sub['genus'].corr(sub['pop']))
    print('genus', g, 'corr genus(age?)', sub['genus'].corr(sub['age']))

