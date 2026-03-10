import pandas as pd

df = pd.read_csv('reading.csv')
for col in ['device','dyslexia']:
    print('\n', col)
    print(df[col].value_counts(dropna=False))

print('\ncorrect_rate counts')
print(df['correct_rate'].value_counts())
print('dyslexia_bin counts')
print(df['dyslexia_bin'].value_counts())
