import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

print(df[['running_time','adjusted_running_time','age','gender','retake_trial','num_words']].describe())
print('corr running_time with adjusted_running_time', df['running_time'].corr(df['adjusted_running_time']))
print('corr running_time with age', df['running_time'].corr(df['age']))
print('corr running_time with gender', df['running_time'].corr(df['gender']))
print('corr running_time with retake_trial', df['running_time'].corr(df['retake_trial']))
print('corr running_time with num_words', df['running_time'].corr(df['num_words']))

# check if running_time ~ retake_trial/age
wpm_from_age = df['retake_trial'] / (df['age']/60000)
print('wpm_from_age summary', wpm_from_age.describe())
print('corr wpm_from_age with running_time', wpm_from_age.corr(df['running_time']))

# check if running_time is time per word maybe
ms_per_word = df['age']/df['retake_trial']
print('ms_per_word summary', ms_per_word.describe())
print('corr ms_per_word with running_time', ms_per_word.corr(df['running_time']))
