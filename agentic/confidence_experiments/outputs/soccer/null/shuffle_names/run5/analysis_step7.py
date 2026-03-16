import pandas as pd

_df = pd.read_csv('soccer.csv')

# pick a player with many rows
player = _df['photoID'].value_counts().idxmax()
print('example player', player, 'rows', _df['photoID'].value_counts().max())
print(_df[_df['photoID']==player][['photoID','goals','rater1','nExp']].head(10))

# show unique rater1 values for that player
print('unique rater1', _df[_df['photoID']==player]['rater1'].unique())
print('unique nExp', _df[_df['photoID']==player]['nExp'].unique())

