import pandas as pd

df = pd.read_csv('reading.csv')

print(pd.crosstab(df['Flesch_Kincaid'], df['correct_rate'], dropna=False))
print('\nFlesch_Kincaid unique', sorted(df['Flesch_Kincaid'].dropna().unique()))
print('correct_rate mean', df['correct_rate'].mean())
print('dyslexia_bin mean', df['dyslexia_bin'].mean())
