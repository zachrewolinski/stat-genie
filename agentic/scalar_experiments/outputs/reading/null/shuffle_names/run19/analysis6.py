import pandas as pd


df = pd.read_csv('reading.csv')
print(pd.crosstab(df['correct_rate'], df['img_width']))
