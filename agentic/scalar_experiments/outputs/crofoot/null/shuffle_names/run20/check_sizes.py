import pandas as pd
import numpy as np

df = pd.read_csv('crofoot.csv')

# check if f_other equals dist_focal + other (males + females for focal)
print('f_other - (dist_focal+other) summary:')
print((df['f_other'] - (df['dist_focal'] + df['other'])).describe())

# check if win equals focal + f_focal (males + females for other)
print('win - (focal+f_focal) summary:')
print((df['win'] - (df['focal'] + df['f_focal'])).describe())

# check if any exact matches
print('f_other exact matches', ((df['f_other'] == df['dist_focal'] + df['other']).mean()))
print('win exact matches', ((df['win'] == df['focal'] + df['f_focal']).mean()))

