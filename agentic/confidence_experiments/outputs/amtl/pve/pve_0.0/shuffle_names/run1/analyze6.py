import pandas as pd

df = pd.read_csv('amtl.csv')

between = ((df['genus'] >= 0) & (df['genus'] <= df['age'])).mean()
print('Proportion genus between 0 and age:', between)

print('genus negatives', (df['genus']<0).mean())
