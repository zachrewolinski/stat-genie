import pandas as pd


df = pd.read_csv('soccer.csv')

df['skin_tone'] = df[['rater1','rater2']].mean(axis=1)

sample = df[df['playerShort'] == 'aaron-hughes'][['playerShort','rater1','rater2','skin_tone']].head(10)
print(sample)
