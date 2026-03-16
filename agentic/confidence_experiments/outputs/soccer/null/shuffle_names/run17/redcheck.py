import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

for col in ['yellowCards','meanExp']:
    s = df[col]
    print(col, 'count>0', (s>0).sum(), 'max', s.max(), 'mean', s.mean())

# check overlap
both = ((df['yellowCards']>0) & (df['meanExp']>0)).sum()
print('both>0', both)

# check correlation
print('corr', df[['yellowCards','meanExp']].corr().iloc[0,1])

# check if one is subset (e.g., meanExp == 1 when yellowCards>0)
print('yellowCards>0 & meanExp==0', ((df['yellowCards']>0) & (df['meanExp']==0)).sum())
print('meanExp>0 & yellowCards==0', ((df['meanExp']>0) & (df['yellowCards']==0)).sum())

# check max combinations
print('unique pairs', df[['yellowCards','meanExp']].drop_duplicates().sort_values(['yellowCards','meanExp']).head(20))
