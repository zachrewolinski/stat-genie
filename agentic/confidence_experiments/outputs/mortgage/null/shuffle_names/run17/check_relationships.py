import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
# check if deny and accept are complements
comp = (df['deny'] + df['accept'])
print('deny+accept unique:', comp.unique())
print('matches complement', ((df['deny'] == 1 - df['accept']) | (df['deny'].isna()) | (df['accept'].isna())).mean())
print(df[['deny','accept']].head())

# correlation
print('corr', df[['deny','accept']].corr())

# female vs deny/accept
for outcome in ['deny','accept']:
    ct = pd.crosstab(df['female'], df[outcome])
    print(outcome, ct)
    # proportion outcome=1 by female
    print(outcome, 'P(outcome=1 | female=1)', ct.loc[1,1]/ct.loc[1].sum())
    print(outcome, 'P(outcome=1 | female=0)', ct.loc[0,1]/ct.loc[0].sum())
