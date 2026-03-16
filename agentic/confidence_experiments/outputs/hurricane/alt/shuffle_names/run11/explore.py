import pandas as pd

df = pd.read_csv('hurricane.csv')
# Katrina row
print('Katrina row:')
print(df[df['alldeaths'].str.contains('Katrina', case=False, na=False)])

print('\nTop deaths (name column):')
print(df.sort_values('name', ascending=False).head(5))

print('\nTop damage? (elapsedyrs column):')
print(df.sort_values('elapsedyrs', ascending=False).head(5))

print('\nColumn stats:')
print('category min/max', df['category'].min(), df['category'].max())
print('masfem min/max', df['masfem'].min(), df['masfem'].max())
print('ind min/max', df['ind'].min(), df['ind'].max())
print('year min/max', df['year'].min(), df['year'].max())
