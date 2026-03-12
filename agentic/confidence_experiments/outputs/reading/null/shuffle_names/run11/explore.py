import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

# helper
print('columns', df.columns.tolist())

# check which columns are likely categorical disguised
for col in df.columns:
    if df[col].dtype==object:
        print('\n',col, 'nunique', df[col].nunique(), 'sample', df[col].dropna().unique()[:5])

# numeric summary
num_cols=[c for c in df.columns if df[c].dtype!=object]
print('\nnum cols', num_cols)

# compute correlations among numeric columns
corr=df[num_cols].corr()
print('\nTop correlations with running_time:')
print(corr['running_time'].sort_values(ascending=False).head(10))

print('\nTop correlations with adjusted_running_time:')
print(corr['adjusted_running_time'].sort_values(ascending=False).head(10))

print('\nTop correlations with age:')
print(corr['age'].sort_values(ascending=False).head(10))

# check if running_time ~ num_words/adjusted_running_time
sub=df[['running_time','num_words','adjusted_running_time','age','gender']].dropna()
sub['wpm_from_adjusted'] = sub['num_words'] / (sub['adjusted_running_time']/60000)
sub['wpm_from_age'] = sub['num_words'] / (sub['age']/60000)
print('\nRunning_time vs wpm_from_adjusted corr', sub['running_time'].corr(sub['wpm_from_adjusted']))
print('Running_time vs wpm_from_age corr', sub['running_time'].corr(sub['wpm_from_age']))

# check distributions
print('\nRunning_time quantiles', df['running_time'].quantile([0.01,0.05,0.5,0.95,0.99]))
print('adjusted_running_time quantiles', df['adjusted_running_time'].quantile([0.01,0.05,0.5,0.95,0.99]))
print('age quantiles', df['age'].quantile([0.01,0.05,0.5,0.95,0.99]))
print('gender quantiles', df['gender'].quantile([0.01,0.05,0.5,0.95,0.99]))

# check if retake_trial values match num_words? compare unique values of retake_trial and num_words
print('\nretake_trial unique', sorted(df['retake_trial'].unique())[:10], 'max', df['retake_trial'].max())
print('num_words unique sample', sorted(df['num_words'].unique())[:10], 'max', df['num_words'].max())

