import pandas as pd
import numpy as np

df=pd.read_csv('affairs.csv')

for cont in ['education','age']:
    for cat in ['rating','affairs','yearsmarried','rownames']:
        corr=df[cont].corr(df[cat])
        print(cont, 'vs', cat, corr)
