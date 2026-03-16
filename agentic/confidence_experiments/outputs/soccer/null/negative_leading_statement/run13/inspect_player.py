import pandas as pd

player = 'aaron-ramsey'

df = pd.read_csv('soccer.csv')
sub = df[df['playerShort'] == player][['rater1','rater2','photoID']].drop_duplicates()
print(sub.head(10))
print('unique combos', len(sub))
