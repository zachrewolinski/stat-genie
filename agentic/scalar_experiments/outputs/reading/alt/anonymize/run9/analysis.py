import pandas as pd
import json

info = json.load(open('info.json'))
print(info['research_questions'])

df = pd.read_csv('reading.csv')
print(df.head())
print(df.describe(include='all'))
print(df.columns)
