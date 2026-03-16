import pandas as pd
import numpy as np

df=pd.read_csv('reading.csv')

# treat retake_trial as num_words candidate
sub=df[['running_time','adjusted_running_time','age','gender','retake_trial','num_words']].dropna()
sub['wpm_from_retake']=sub['retake_trial'] / (sub['adjusted_running_time']/60000)
sub['wpm_from_retake_age']=sub['retake_trial'] / (sub['age']/60000)
sub['wpm_from_num_words']=sub['num_words'] / (sub['adjusted_running_time']/60000)

for name in ['wpm_from_retake','wpm_from_retake_age','wpm_from_num_words']:
    print(name, sub['running_time'].corr(sub[name]))

print('retake_trial unique', sorted(df['retake_trial'].unique()))
print('num_words unique count', df['num_words'].nunique())

