import pandas as pd
_df = pd.read_csv('mortgage.csv')
print('deny==self_employed', (_df['deny']==_df['self_employed']).mean())
print('deny==1-self_employed', (_df['deny']==(1-_df['self_employed'])).mean())
print('accept==self_employed', (_df['accept']==_df['self_employed']).mean())
print('accept==1-self_employed', (_df['accept']==(1-_df['self_employed'])).mean())
print('accept==deny', (_df['accept']==_df['deny']).mean())
print('accept==1-deny', (_df['accept']==(1-_df['deny'])).mean())
