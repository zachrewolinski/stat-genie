import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')

for cand_aff in ['age', 'education', 'rating', 'affairs']:
    for cand_rating in ['rating', 'affairs']:
        corr = np.corrcoef(df[cand_aff], df[cand_rating])[0,1]
        print(f"corr({cand_aff},{cand_rating})={corr:.3f}")
