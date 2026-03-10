import pandas as pd

df = pd.read_csv('reading.csv')
num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()
print('Correlation of adjusted_running_time with retake_trial:', corr.loc['adjusted_running_time','retake_trial'])
print('Correlation of running_time with retake_trial:', corr.loc['running_time','retake_trial'])
print('Correlation of age with retake_trial:', corr.loc['age','retake_trial'])
print('Correlation of gender with retake_trial:', corr.loc['gender','retake_trial'])
