import pandas as pd

df = pd.read_csv('crofoot.csv')

print('f_other - dist_focal equals other?')
print(((df['f_other'] - df['dist_focal']) == df['other']).mean())
print((df['f_other'] - df['dist_focal'] - df['other']).describe())

print('win - focal equals f_focal?')
print(((df['win'] - df['focal']) == df['f_focal']).mean())
print((df['win'] - df['focal'] - df['f_focal']).describe())

