import pandas as pd

df=pd.read_csv('crofoot.csv')
for id_col in ['n_other','dist_other']:
    print('ID col', id_col)
    for col in ['f_other','win','dist_focal','focal','other','f_focal']:
        uniq=df.groupby(id_col)[col].nunique()
        print(col, 'unique per id (min,max)', uniq.min(), uniq.max())
    print()
