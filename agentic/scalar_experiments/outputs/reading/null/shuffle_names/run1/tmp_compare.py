import pandas as pd

df = pd.read_csv('reading.csv')

print('Flesch_Kincaid mean by dyslexia_bin')
print(df.groupby('dyslexia_bin')['Flesch_Kincaid'].mean())

print('\nFlesch_Kincaid mean by correct_rate')
print(df.groupby('correct_rate')['Flesch_Kincaid'].mean())

print('\ncorrect_rate mean by dyslexia_bin')
print(df.groupby('dyslexia_bin')['correct_rate'].mean())

print('\ndyslexia_bin mean by correct_rate')
print(df.groupby('correct_rate')['dyslexia_bin'].mean())
