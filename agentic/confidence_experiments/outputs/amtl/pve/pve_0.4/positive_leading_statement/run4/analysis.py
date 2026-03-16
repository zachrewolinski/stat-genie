import pandas as pd
import json

info = json.load(open('info.json'))
print('question:', info['research_questions'][0])

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df.describe(include='all').transpose().head(10))
print('num rows', len(df))
print('num_amtl unique sample', df['num_amtl'].head())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets unique', df['sockets'].unique()[:10])
print('genus counts', df['genus'].value_counts())
print('tooth_class counts', df['tooth_class'].value_counts())
print('prob_male unique', df['prob_male'].unique())
