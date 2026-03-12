import pandas as pd

df = pd.read_csv('amtl.csv')
within = ((df['genus']>=0) & (df['genus']<=df['age'])).mean()
print('genus within 0-age proportion', within)
within_num = ((df['num_amtl']>=0) & (df['num_amtl']<=df['age'])).mean()
print('num_amtl within 0-age proportion', within_num)
