import pandas as pd

cols = ['f_other','win','dist_focal','focal','other','f_focal']

df = pd.read_csv('crofoot.csv')
print(df[cols].corr())
