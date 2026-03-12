import pandas as pd


df = pd.read_csv('reading.csv')

# ms per word

df['ms_per_word'] = df['feature5'] / df['feature7']
df['ms_per_word_total'] = df['feature4'] / df['feature7']

print('corr feature20 vs ms_per_word', df[['feature20','ms_per_word']].corr().iloc[0,1])
print('corr feature20 vs ms_per_word_total', df[['feature20','ms_per_word_total']].corr().iloc[0,1])

print(df['ms_per_word'].describe())
print(df['ms_per_word_total'].describe())

# check if feature20 approx ms_per_word? ratio
print('median ratio feature20/ms_per_word', (df['feature20']/df['ms_per_word']).median())
print('median ratio feature20/ms_per_word_total', (df['feature20']/df['ms_per_word_total']).median())

